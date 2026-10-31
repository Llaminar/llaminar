/**
 * @file MPIOwnedSharedBorrowRace.c
 * @brief Deterministic test-only replacement at actual root authentication/use edges.
 *
 * All mutation is restricted to the disposable run's original shared directory.
 * The observer restores both exact original inodes before the guardian retires;
 * a foreign/nonempty replacement is retained rather than removed by a pattern.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
static int armed=0,swapped=0;
static char original[4096],saved[4096];
static struct stat original_id,replacement_id;
/** @brief Replace only this run's observed root once; every diagnostic inode is retained. */
static void swap_root(void)
{
    armed=0;
    const char *root=getenv("OMPI_MCA_btl_vader_backing_directory");
    if(!root || snprintf(original,sizeof(original),"%s",root)>=(int)sizeof(original) ||
       snprintf(saved,sizeof(saved),"%s-borrow-original-%ld",root,(long)getpid())>=(int)sizeof(saved) ||
       lstat(original,&original_id) || !S_ISDIR(original_id.st_mode) || original_id.st_uid!=getuid() ||
       rename(original,saved) || mkdir(original,0700) || lstat(original,&replacement_id))_exit(126);
    swapped=1;
}
/** @brief Select managed final-marker or native exact-open checkpoint, after ordinary entry. */
void llaminarTestSharedBorrowArm(int mode) { armed=mode; }
/** @return Whether the actual injection ran, separate from the production result. */
int llaminarTestSharedBorrowSwapped(void) { return swapped; }
/** @brief Restore only the two exact observed directory identities. */
int llaminarTestSharedBorrowRestore(void)
{
    armed=0;if(!swapped)return 0;
    struct stat a,b;
    if(lstat(original,&a) || lstat(saved,&b) ||
       a.st_dev!=replacement_id.st_dev || a.st_ino!=replacement_id.st_ino || a.st_uid!=replacement_id.st_uid ||
       b.st_dev!=original_id.st_dev || b.st_ino!=original_id.st_ino || b.st_uid!=original_id.st_uid ||
       rmdir(original) || rename(saved,original))return -1;
    swapped=0;return 0;
}
ssize_t read(int fd,void *buffer,size_t size)
{
    ssize_t (*native)(int,void*,size_t)=dlsym(RTLD_NEXT,"read");if(!native)_exit(126);
    const ssize_t count=native(fd,buffer,size);
    if(armed==1 && count>13 && !memcmp(buffer,"guardian_pid=",13)) {
        const char *temporary=getenv("TMPDIR");char marker[4096];struct stat a,b;
        if(temporary && snprintf(marker,sizeof(marker),"%s/.owner",temporary)<(int)sizeof(marker) &&
           !fstat(fd,&a) && !lstat(marker,&b) && a.st_dev==b.st_dev && a.st_ino==b.st_ino)swap_root();
    }
    return count;
}
int open(const char *path,int flags,...)
{
    int (*native)(const char*,int,...)=dlsym(RTLD_NEXT,"open");if(!native)_exit(126);
    mode_t mode=0;
    if((flags&O_CREAT) || ((flags&O_TMPFILE)==O_TMPFILE)) {va_list args;va_start(args,flags);mode=va_arg(args,int);va_end(args);}
    if(armed==2) {
        const char *root=getenv("OMPI_MCA_btl_vader_backing_directory");
        if(root && !strcmp(path,root))swap_root();
    }
    return native(path,flags,mode);
}
