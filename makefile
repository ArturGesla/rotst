INC=-I../eigen

rotst: main.cpp rotst.cpp rotst-2.cpp lean_vtk.cpp
	g++ $(INC) -o rotst main.cpp rotst.cpp rotst-2.cpp lean_vtk.cpp -I.

clean:
	rm rotst

o3: main.cpp rotst.cpp rotst-2.cpp lean_vtk.cpp
	g++ $(INC) -O3 -o rotst main.cpp rotst.cpp rotst-2.cpp lean_vtk.cpp -I.
        

