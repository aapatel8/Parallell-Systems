/* 'C' program showing usage of persistent memory functions */

#include <stdio.h>
#include <string.h>

/* use JEMALLOC_MANGLE if: configure --with-jemalloc-prefix=je */
/* #define JEMALLOC_MANGLE */
#include "jemalloc/jemalloc.h"

#define BACK_FILE "/tmp/app.back" /* Note: different backup and mmap files */
#define MMAP_FILE "/tmp/app.mmap"
#define MMAP_SIZE ((size_t)1 << 30)

typedef struct {
/* ... */ 
} home_st;

PERM home_st *home; /* use PERM to mark home as persistent */

int main(int argc, char *argv[])
{
int do_restore = argc > 1 && strcmp("-r", argv[1]) == 0;
char *mode = (do_restore) ? "r+" : "w+";

/* call perm() and open() before malloc() */
perm(PERM_START, PERM_SIZE);
mopen(MMAP_FILE, mode, MMAP_SIZE);
bopen(BACK_FILE, mode);
if (do_restore) {
restore();
} else {
home = (home_st *)malloc(sizeof(home_st));
/* initialize home struct... */
mflush(); backup(); 
}

for (;/* each step */;) {
/* Application_Step(); */ 
backup();
}

free(home); 
mclose(); 
bclose(); 
return(0); 
}