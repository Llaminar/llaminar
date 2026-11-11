/**
 * @file NativeCPUEntryExecFault.c
 * @brief Isolated syscall/initial-worker controls for native CPU entry lifecycle tests.
 *
 * Only the explicit /proc/self/exe startup call is intercepted. Normal exec
 * calls keep their native implementation; no production inference path uses
 * this fixture. Audit paths belong exclusively to the invoking test case.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static pthread_t worker;
static int worker_created;
static atomic_int stop_worker;

/** @brief Keep one observable pre-entry worker alive until fixture teardown. */
static void *initial_worker(void *unused)
{
    (void)unused;
    const struct timespec delay={0,1000000};
    while(!atomic_load(&stop_worker))nanosleep(&delay,NULL);
    return NULL;
}
/** @brief Admit an explicitly requested pre-entry worker for the rejection control. */
__attribute__((constructor)) static void prepare_worker(void)
{
    const char *enabled=getenv("LLAMINAR_TEST_NATIVE_ENTRY_WORKER");
    if(enabled && strcmp(enabled,"1")==0)
    {
        if(pthread_create(&worker,NULL,initial_worker,NULL)!=0)_exit(124);
        worker_created=1;
    }
}
/** @brief Join every worker created by this fixture before normal process retirement. */
__attribute__((destructor)) static void retire_worker(void)
{
    if(worker_created)
    {
        atomic_store(&stop_worker,1);
        if(pthread_join(worker,NULL)!=0)_exit(124);
    }
}
/**
 * @brief Intercept only the original-entry native exec for an explicit test action.
 * @param path Exact executable pathname supplied by production.
 * @param argv Original application arguments supplied by production.
 * @return Native exec result, or -1 with the injected failure after auditing.
 */
int execv(const char *path,char *const argv[])
{
    typedef int (*native_exec)(const char *,char *const[]);
    native_exec original=(native_exec)dlsym(RTLD_NEXT,"execv");
    if(!original){errno=ENOSYS;return -1;}
    const char *action=getenv("LLAMINAR_TEST_NATIVE_EXEC_ACTION");
    if(!action || strcmp(path,"/proc/self/exe")!=0)return original(path,argv);
    const char *audit=getenv("LLAMINAR_TEST_NATIVE_EXEC_AUDIT");
    const char *temporary=getenv("TMPDIR");
    const char *shared=getenv("OMPI_MCA_btl_vader_backing_directory");
    struct stat identity;
    if(!audit || !temporary || !shared || stat(shared,&identity)!=0){errno=EINVAL;return -1;}
    char pending[4096];
    const int pending_size=snprintf(pending,sizeof(pending),"%s.pending",audit);
    if(pending_size<0 || (size_t)pending_size>=sizeof(pending)){errno=ENAMETOOLONG;return -1;}
    const int fd=open(pending,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
    if(fd<0)return -1;
    char record[8192];
    const int bytes=snprintf(record,sizeof(record),"pid=%ld\ntemporary=%s\nshared=%s\nshared_device=%llu\nshared_inode=%llu\n",
        (long)getpid(),temporary,shared,(unsigned long long)identity.st_dev,(unsigned long long)identity.st_ino);
    if(bytes<0 || (size_t)bytes>=sizeof(record)){close(fd);errno=EIO;return -1;}
    size_t written=0;
    while(written<(size_t)bytes)
    {
        const ssize_t count=write(fd,record+written,(size_t)bytes-written);
        if(count<0 && errno==EINTR)continue;
        if(count<=0){close(fd);errno=EIO;return -1;}
        written+=(size_t)count;
    }
    if(close(fd)!=0 || link(pending,audit)!=0 || unlink(pending)!=0)return -1;
    if(strcmp(action,"fail")==0){errno=EACCES;return -1;}
    if(strcmp(action,"pause")==0)for(;;)pause();
    if(strcmp(action,"pass")!=0){errno=EINVAL;return -1;}
    return original(path,argv);
}
