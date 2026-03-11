/**
 * @file Main.cpp
 * @brief Llaminar v2 entry point
 */

#include "app/AppLifecycle.h"

int main(int argc, char *argv[])
{
    llaminar2::AppLifecycle app;
    return app.run(argc, argv);
}
