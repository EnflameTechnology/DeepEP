/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_ALIGN_H_
#define EP_ALIGN_H_

#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))

#define ROUNDUP(x, y) \
    (DIVUP((x), (y))*(y))

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);


/* They are defined here to allow both the host and the kernel to access it.
* To save mmu map num for GCU 300
* reserve first 1k for abort & other uses(op index, etc.) and the next bytes for ras:
* currently the first 1k: uint32_t is used for the comm abort flag,and next for the op index of channel
*
*/
#define MEM_STACK_RESERVE_SIZE (1024 * 8)
#define RAS_MEM_START 1024

#endif
