#include <iostream>
#include <cstdlib>

// #include <unistd.h>

#include "mldata.hpp"
#include "board.hpp"


int main(int argc, char const *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "Usage: read_mldata file limit\n");
        exit(-1);
    }

    FILE *fp = fopen(argv[1], "rb");
    if (!fp) {
        fprintf(stderr, "cannot open file \"%s\"\n", argv[1]);
        exit(-1);
    }

    int limit = atoi(argv[2]);
    printf("limit = %d\n\n", limit);

    int i = 0;
    int retval;
    while (true) {
        entry_t entry;
        retval = fread(&entry, sizeof(entry_t), 1, fp);
        if (retval == 0) {
            break;
        }

        Board board;
        board.set_boards(entry.black_bitboard, entry.white_bitboard, Side::BLACK);
        std::cout << "i=" << i << std::endl;
        std::cout << board;
        std::cout << "side=" << entry.side
            << " action=" << entry.action
            << " Q=" << entry.Q
            << " result=" << entry.result << std::endl;
        float sum = 0;
        for (float posterior : entry.posteriors) {
            std::cout << posterior << " ";
            sum += posterior;
        }
        std::cout << "\nsum=" << sum << std::endl;

        i++;
        if (i == limit) {
            break;
        }
    }

    fclose(fp);

    return 0;
}