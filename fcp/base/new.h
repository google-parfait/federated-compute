#ifndef FCP_BASE_NEW_H_
#define FCP_BASE_NEW_H_

#ifdef FCP_NANOLIBC
// Definitions of placement operator new are needed because nanolibc doesn't
// currently have the <new> header.
inline void* operator new(size_t, void* p) noexcept { return p; }
inline void* operator new[](size_t, void* p) noexcept { return p; }
#else
#include <new>
#endif  // FCP_NANOLIBC

#endif  // FCP_BASE_NEW_H_
