# Makefile principal
# VARIABLES POUVANT ETRE REDEFINIS PAR L'UTILISATEUR
# DIRECTORY OU EST INSTALLE LE PROJET :
PROJECT_ROOT = $(CURDIR)/..
##############################################################
# Compilateur utilise
CXX = g++
# Options pour le debogage et l'optimisation
DEBUGOPT = -g -Wall -pedantic -fopenmp -D_GLIBCXX_DEBUG_PEDANTIC
OPTIMOPT = -O2 -march=native -fopenmp
# Parametres passes au compilateur
CXXOPT = -std=c++11 -fPIC $(DEBUGOPT)

# Ou est installe MPI ( utile que si on utilise autre chose qu'OpenMPI )
MPIROOT    = /usr/lib/openmpi

# Ou se trouve les fichiers d'entete d'OpenGL et les librairies
OPENGLINC      = /usr/local/include
OPENGLLIBROOT  = /usr/local/lib
OPENGLLDFLAGS  = -L$(OPENGLLIBROOT) -lglut -lGLU -lGL

# Si doxygen est installe, il est installe ou ?
DOXYGENEXE = /usr/bin/doxygen
# FIN DES VARIABLES POUVANT ETRE REDEFINIES PAR L'UTLISATEUR

INCPATH= -I$(MPIROOT)/include                \
	     -I$(PROJECT_ROOT)/include

LIBPATH= -L$(OPENGLLIBROOT) -L$(PROJECT_ROOT)/lib 
