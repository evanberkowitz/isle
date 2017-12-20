####
CXX     =mpic++-mpich-clang39
CXXOPT  =-std=c++14 -pedantic -stdlib=libc++
CXXFLAGS=$(CXXOPT) -g -O0

####
INCL=
LIBS=

####
HDRS:=CNS.hpp
OBJS:=

####
CNS: $(OBJS) CNS.o $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) CNS.o $(LIBS)

####
%.o: %.cc $(HDRS)
	$(CXX) $(CXXFLAGS) $(INCL) -c $<

#---- DOC ----
.PHONY: doxy
doxy: $(OBJECTS) $(HDRS)
	doxygen config.doxy

####
.PHONY: clean
clean:
	rm -f $(OBJS)
	rm -f CNS
	rm -rf *.dSYM