EXECS=owlqn

MPICC=mpicc


CFLAGS=-Wall

CPPFLAGS= -std=gnu++11

GLOOG_LIB=-L ../../local/glog/lib/ \
 -L ../../local/gflag/lib/ 

GTEST_LIB=/home/tuan/softwares/googletest-master/build/gtest/googlemock/gtest/libgtest.a

INCPATH=-I ./include/\
 -I ../../local/glog/include/\
 -I ../../local/gflag/include/\
 -I ../../softwares/googletest-master/googletest/include/

OBJS=$(patsubst %.cpp, %.o, $(wildcard ./src/*.cpp))  

.PHONY : clean all

all: ${OBJS} ${EXECS} main_test

clean:
	rm -f ./src/*.o  ${EXECS} main_test ./*.o

%.o:%.cpp 
	${MPICC} ${CPPFLAGS}  -c ${INCPATH}  $^ -o $@ 

${EXECS}: main.cpp ${OBJS}
	${MPICC}  ${CPPFLAGS}  ${INCPATH} ${GLOOG_LIB}  $^ -o $@ -lstdc++ -lpthread -lglog -lgflags -lm

main_ut.o: ./utest/main_test.cpp
	${MPICC}  ${CPPFLAGS}  ${INCPATH} -c ./utest/main_test.cpp -o $@

main_test: ./utest/main_test.cpp ${OBJS}
	${MPICC}  ${CPPFLAGS} ${INCPATH} ${GLOOG_LIB} $^  ${GTEST_LIB} -o $@ -lstdc++ -lpthread -lglog -lgflags  -lm
