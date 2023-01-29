#ifndef __HOOK_CUTABLE_H__
#define __HOOK_CUTABLE_H__

#include "nvbit.h"

#define CU_UUID_CONST static const
#define CU_CHAR(x) (char)((x) & 0xff)
// Define the symbol as exportable to other translation units, and
// initialize the value.  Inner set of parens is necessary because
// "bytes" array needs parens within the struct initializer, which
// also needs parens.  
#define CU_DEFINE_UUID(name, a, b, c, d0, d1, d2, d3, d4, d5, d6, d7)          \
    CU_UUID_CONST CUuuid name =                                                \
    {                                                                          \
      {                                                                        \
        CU_CHAR(a), CU_CHAR((a) >> 8), CU_CHAR((a) >> 16), CU_CHAR((a) >> 24), \
        CU_CHAR(b), CU_CHAR((b) >> 8),                                         \
        CU_CHAR(c), CU_CHAR((c) >> 8),                                         \
        CU_CHAR(d0),                                                           \
        CU_CHAR(d1),                                                           \
        CU_CHAR(d2),                                                           \
        CU_CHAR(d3),                                                           \
        CU_CHAR(d4),                                                           \
        CU_CHAR(d5),                                                           \
        CU_CHAR(d6),                                                           \
        CU_CHAR(d7)                                                            \
      }                                                                        \
    }

CU_DEFINE_UUID(CU_ETID_ToolsRm,
    0x0d180614, 0xd672, 0x47cb, 0xab, 0x6e, 0xb9, 0x63, 0xe5, 0x1b, 0xf4, 0xc6);

CU_DEFINE_UUID(CU_ETID_ToolsModule,
    0xbe3f166e, 0x58b9, 0x4d44, 0x83, 0x5c, 0xe1, 0x82, 0xaf, 0xf1, 0x99, 0x1e);

#endif