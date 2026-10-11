#include "assignment.h"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        if (options.help) {
            printUsage(argv[0]);
            return 0;
        }
        runAssignment(options);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
