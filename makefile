INCL=-I./library/inc/
INCEXT=-I../eigen/  -I./external/lean_vtk/
INC=$(INCEXT) $(INCL)

SRCEXT=./external/lean_vtk/lean_vtk.cpp
#SRC=./drivers/main.cpp ./library/src/* ./library/src/rotst-2.cpp $(SRCEXT)
SRC=./drivers/main.cpp ./library/src/* $(SRCEXT)

rotst: $(SRC)
	g++ -o rotst $(SRC) $(INC)

clean:
	rm rotst

#o3: main.cpp rotst.cpp rotst-2.cpp lean_vtk.cpp
#	g++ $(INC) -O3 -o rotst main.cpp rotst.cpp rotst-2.cpp lean_vtk.cpp -I.


o3: $(SRC)
	g++ -O3 -o rotst $(SRC) $(INC)

