#define DETECTOR_VARIANT 3  //see below

//CACHE_STRIDE_SHIFT        detector stride for every value of byte (>= 128)
//CACHE_LINE_OFFSET_SHIFT   offset to avoid cache way conflicts
//CACHE_LINE_SPLIT          offset within the cacheline (8 bytes fetched from this offset)

#if DETECTOR_VARIANT == 0
#define CACHE_STRIDE_SHIFT          9   //512
#define CACHE_LINE_OFFSET_SHIFT     0
#define CACHE_LINE_SPLIT            60

#elif DETECTOR_VARIANT == 1
#define CACHE_STRIDE_SHIFT          12  //4096
#define CACHE_LINE_OFFSET_SHIFT     0
#define CACHE_LINE_SPLIT            0

#elif DETECTOR_VARIANT == 2
#define CACHE_STRIDE_SHIFT          14  //16384
#define CACHE_LINE_OFFSET_SHIFT     6   //64     avoid problems with L1 TLB / bank conflicts?
#define CACHE_LINE_SPLIT            0

#elif DETECTOR_VARIANT == 3
#define CACHE_STRIDE_SHIFT          15  //32768
#define CACHE_LINE_OFFSET_SHIFT     7   //128
#define CACHE_LINE_SPLIT            60

#elif DETECTOR_VARIANT == 4
#define CACHE_STRIDE_SHIFT          15  //32768
#define CACHE_LINE_OFFSET_SHIFT     6   //64
#define CACHE_LINE_SPLIT            60

#endif

#define DUMP_DETECTION_ERRORS 1  //show error while calibrating
#define USE_RAW_TIMERS        0  //read hi-res timers directly, if possible
#define EVICT_CACHE			  0  //evict cache without cache flush instructions

//Best for A9 [evict = 16MB]
//16384 64  0  0-4         100%          tlb_miss
// 4096  0  0  0-3         99.6 .. 100%
//  512  0 60  0-3 ticks   99.2 .. 100%  cl_cross

//Best for A11 [evict =32MB]
//32768 64  60  0-4        98  tlb_miss + cl_cross + bank_conflict?
//32768 0 16380 1-4        97  tlb_miss + page_cross
//32768 128 60  0-3        94  tlb_miss + cl_cross

//A7,A11 - #3
//A9 - #2
