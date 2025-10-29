#pragma once
#include <iostream>

#define DEBUG 0
#if DEBUG > 0
    #define DEBUG_PRINT(fmt, ...) printf("[Debug] " fmt, ##__VA_ARGS__)
    #if DEBUG > 1
        #define WARN_PRINT(fmt, ...) printf("[Warn] " fmt, ##__VA_ARGS__)
        #if DEBUG > 2
            #define ERROR_PRINT(fmt, ...) printf("[Error] " fmt, ##__VA_ARGS__)
        #else 
            #define ERROR_PRINT(fmt, ...) ((void)0)
        #endif // ERROR
    #else
        #define ERROR_PRINT(fmt, ...) ((void)0)
        #define WARN_PRINT(fmt, ...) ((void)0)
    #endif // WARN
#else
    #define ERROR_PRINT(fmt, ...) ((void)0)
    #define WARN_PRINT(fmt, ...) ((void)0)
    #define DEBUG_PRINT(fmt, ...) ((void)0)
#endif // DEBUG