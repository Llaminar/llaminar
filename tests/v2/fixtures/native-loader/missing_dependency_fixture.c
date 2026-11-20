/**
 * @file missing_dependency_fixture.c
 * @brief Native loader fixture whose one dependency is owned and absent.
 *
 * The object exports a probe with no imports of its own; linking against
 * the removed stub records exactly one NEEDED entry,
 * libllaminar-absent-dependency.so.9, which no installed environment
 * provides. Loading it exercises the loader's missing-dependency path with
 * a contract this tree owns, independent of host architecture or libc.
 */
int llaminar_native_loader_missing_dependency_probe(void)
{
    return 7;
}
