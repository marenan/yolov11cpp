// Debug.hpp
#ifndef DEBUG_HPP
#define DEBUG_HPP


// Include necessary libraries
#include <iostream>
#include "./tools/Config.hpp" // Include the config file to access the flags

#ifdef DEBUG_MODE
#define DEBUG_PRINT(x) std::cout << x << std::endl;
#else
#define DEBUG_PRINT(x)
#endif

#endif // DEBUG_HPP
