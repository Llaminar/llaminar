/**
 * @file loadable_fixture.c
 * @brief Native loader fixture proving the positive presence path.
 *
 * The object is a complete, dependency-light shared library owned by this
 * tree: loading it must succeed through the same shared native probe the
 * production driver-presence authority uses.
 */
int llaminar_native_loader_loadable_probe(void)
{
    return 11;
}
