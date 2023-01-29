/**
 *  @brief utils.h file
 *  contains public utils for test_image
 **/

#ifndef TEST_IMAGE_PUB_H
#define TEST_IMAGE_PUB_H

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// Read file to buffer
class BufferFile {
 public :
    std::string file_path_;
    int length_;
    char* buffer_;

    explicit BufferFile(std::string file_path, std::string type = "bin")
    :file_path_(file_path) {

        std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
        if (!ifs) {
            std::cerr << "Can't open the file. Please check " << file_path << ". \n";
            length_ = 0;
            buffer_ = NULL;
            return;
        }

        ifs.seekg(0, std::ios::end);
        length_ = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        // std::cout << file_path.c_str() << " ... "<< length_ << " bytes" << std::endl;

        if (type == "txt") {
          buffer_ = new char[sizeof(char) * (length_ + 1)];
          ifs.read(buffer_, length_);
          buffer_[length_] = 0;
        } else {
          buffer_ = new char[sizeof(char) * length_];
          ifs.read(buffer_, length_);
        }
        ifs.close();
    }

    int GetLength() {
        return length_;
    }
    char* GetBuffer() {
        return buffer_;
    }

    ~BufferFile() {
        if (buffer_) {
          delete[] buffer_;
          buffer_ = NULL;
        }
    }
};

#endif
