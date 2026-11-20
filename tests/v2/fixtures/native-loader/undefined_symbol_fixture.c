/**
 * @file undefined_symbol_fixture.c
 * @brief Native loader fixture with one owned unresolvable import.
 *
 * The object imports llaminar_test_undefined_import_v1, which nothing in
 * this tree or any system environment defines. A plain shared link leaves
 * the import undefined by design, so loading the fixture with the binding's
 * exact resolve-everything flags exercises the loader's undefined-symbol
 * path with a contract this tree owns, independent of host architecture.
 */
extern int llaminar_test_undefined_import_v1(void);

int llaminar_native_loader_undefined_symbol_probe(void)
{
    return llaminar_test_undefined_import_v1();
}
