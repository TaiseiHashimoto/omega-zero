BUILD_DIR := ./build

TARGET := reversi
TARGET2 := play

SRCS := board.cpp
SRCS += node.cpp
SRCS += mcts.cpp
SRCS += network.cpp
SRCS += mldata.cpp
SRCS += misc.cpp

OBJS := $(SRCS:%.cpp=$(BUILD_DIR)/%.o)
DEPS := $(SRCS:%.cpp=$(BUILD_DIR)/%.d)

CXX := g++
CPPFLAGS := -g -O0 -Wall -MMD -MP
OBJFLAGS := 
LDFLAGS := -pthread


main: $(BUILD_DIR)/$(TARGET)
	rm -f core
play: $(BUILD_DIR)/$(TARGET2)
	rm -f core

$(BUILD_DIR)/%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/$(TARGET): $(BUILD_DIR)/main.o $(OBJS)
	$(CXX) $(OBJFLAGS) $^ -o $@ $(LDFLAGS)
$(BUILD_DIR)/$(TARGET2): $(BUILD_DIR)/play.o $(OBJS)
	$(CXX) $(OBJFLAGS) $^ -o $@ $(LDFLAGS)

-include $(DEPS)


.PHONY: clean rmcore
clean:
	$(RM) $(BUILD_DIR)/*
