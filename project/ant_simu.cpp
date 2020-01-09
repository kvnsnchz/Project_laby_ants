#include <vector>
#include <iostream>
#include <random>
#include "labyrinthe.hpp"
#include "ant.hpp"
#include "pheromone.hpp"
# include "gui/context.hpp"
# include "gui/colors.hpp"
# include "gui/point.hpp"
# include "gui/segment.hpp"
# include "gui/triangle.hpp"
# include "gui/quad.hpp"
# include "gui/event_manager.hpp"
# include "display.hpp"
#include "omp.h"
#include <mpi.h>

void advance_time( const labyrinthe& land, pheromone& phen, 
                   const position_t& pos_nest, const position_t& pos_food,
                   std::vector<ant>& ants, std::size_t& cpteur )
{
    #pragma omp parallel for
    for ( size_t i = 0; i < ants.size(); ++i ){
        ants[i].advance(phen, land, pos_food, pos_nest, cpteur);
    }
    phen.do_evaporation();
    phen.update();
}

int main(int nargs, char* argv[])
{   
    int nbp, rank;
    const dimension_t dims{64,128};// Dimension du labyrinthe
    const double alpha=0.97; // Coefficient de chaos
    //const double beta=0.9999; // Coefficient d'évaporation
    const double beta=0.999; // Coefficient d'évaporation 
    const int nb_ants = 2*dims.first*dims.second; // Nombre de fourmis
    const std::size_t life = int(dims.first*dims.second);

    // Location du nid
    position_t pos_nest{dims.first/2,dims.second/2};
    // Location de la nourriture
    position_t pos_food{dims.first-1,dims.second-1};

    labyrinthe laby(dims);

    MPI_Init(&nargs, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nbp);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 1){
        const double eps = 0.75;  // Coefficient d'exploration
                            
        
        // Définition du coefficient d'exploration de toutes les fourmis.
        ant::set_exploration_coef(eps);
        // On va créer toutes les fourmis dans le nid :
        std::vector<ant> ants;
        ants.reserve(nb_ants);
        for ( size_t i = 0; i < nb_ants; ++i )
            ants.emplace_back(pos_nest, life);
        // On crée toutes les fourmis dans la fourmilière.
        pheromone phen(laby.dimensions(), pos_food, pos_nest, alpha, beta);

        //MPI_Ssend();

    }
    
       
    if(rank == 0){
        std::vector<ant> ants;
        for ( size_t i = 0; i < nb_ants; ++i )
            ants.emplace_back(pos_nest, life);
        pheromone phen(laby.dimensions(), pos_food, pos_nest, alpha, beta);

        gui::context graphic_context(nargs, argv);
        gui::window& win =  graphic_context.new_window(h_scal*laby.dimensions().second,h_scal*laby.dimensions().first+266);
    
        display_t displayer( laby, phen, pos_nest, pos_food, ants, win );
        size_t food_quantity = 0;
        
        gui::event_manager manager;
        manager.on_key_event(int('q'), [] (int code) { MPI_Finalize(); exit(0); });
        manager.on_display([&] { displayer.display(food_quantity); win.blit(); });
        manager.on_idle([&] () { 
            //advance_time(laby, phen, pos_nest, pos_food, ants, food_quantity);

            displayer.display(food_quantity); 
            win.blit(); 
        });
        manager.loop();
    }
    
    MPI_Finalize();
    return 0;
}