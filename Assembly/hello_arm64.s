.section    __TEXT,__text,regular,pure_instructions
.globl  _main

_main:
    // write(1, message, length)
    mov x0, 1              // stdout file descriptor
    ldr x1, =message       // pointer to message
    mov x2, 14             // message length
    mov x16, 0x2000004     // syscall number for write
    svc 0

    // exit(0)
    mov x0, 0              // exit code
    mov x16, 0x2000001     // syscall number for exit
    svc 0

.section    __TEXT,__cstring
message:
    .asciz "Hello, World!\n"
