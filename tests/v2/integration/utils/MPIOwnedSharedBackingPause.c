/**
 * @file MPIOwnedSharedBackingPause.c
 * @brief Test-only pause after actual root-contained ShmemSpin backing creation.
 *
 * The actual production ftruncate executes first. Only the exact ShmemSpin
 * basename under this process's published root is observed. The harness retains
 * original kernel/inode witnesses and interrupts the creator before publication.
 * No production phase or cleanup algorithm is replaced by this observer.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int ftruncate(int descriptor,off_t length)
{
    int (*native)(int,off_t)=dlsym(RTLD_NEXT,"ftruncate");
    if(!native)_exit(126);
    const int result=native(descriptor,length);
    if(result)return result;
    char link[64],path[4096];
    snprintf(link,sizeof(link),"/proc/self/fd/%d",descriptor);
    const ssize_t bytes=readlink(link,path,sizeof(path)-1);
    if(bytes<0)return result;
    path[bytes]='\0';
    const char *basename=strrchr(path,'/');
    const char *shared=getenv("OMPI_MCA_btl_vader_backing_directory");
    if(!basename || strncmp(basename+1,"llaminar_shmem_ar_",strlen("llaminar_shmem_ar_")) ||
       !shared || (size_t)(basename-path)!=strlen(shared) || strncmp(path,shared,strlen(shared)))return result;
    const char *record=getenv("LLAMINAR_SHARED_BACKING_PAUSE_REPORT"),*temporary=getenv("TMPDIR");
    struct stat file;
    if(!record || !temporary || fstat(descriptor,&file))_exit(126);
    char pending[4096];
    if(snprintf(pending,sizeof(pending),"%s.pending",record)>=(int)sizeof(pending))_exit(126);
    const int report=open(pending,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
    if(report<0)_exit(126);
    dprintf(report,"{\"pid\":%ld,\"path\":\"%s\",\"temporary\":\"%s\",\"shared\":\"%s\",\"dev\":%llu,\"ino\":%llu,\"uid\":%lu,\"bytes\":%lld}\n",
        (long)getpid(),path,temporary,shared,(unsigned long long)file.st_dev,(unsigned long long)file.st_ino,
        (unsigned long)file.st_uid,(long long)file.st_size);
    if(close(report) || rename(pending,record))_exit(126);
    for(;;)pause();
}
