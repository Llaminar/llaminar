/**
 * @file missing_dependency_fixture.c
 * @brief Native loader fixture whose one dependency is owned and absent.
 *
 * The probe has no imports of its own. Linking against the stub records
 * libllaminar-absent-dependency.so.9 as a NEEDED entry, while the stub's actual
 * filename differs and stays in the build tree. Loading the fixture exercises
 * the missing-dependency path independently of host architecture or libc.
 */
int llaminar_native_loader_missing_dependency_probe(void)
{
    return 7;
}
