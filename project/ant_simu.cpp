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
#include <chrono>
#include <omp.h>

std::chrono::time_point<std::chrono::system_clock> start[5], end[5];
std::chrono::duration<double> elapsed_seconds[5];

void print_time_execution(){
    std::cout << "Time of function advance: " << elapsed_seconds[1].count() << std::endl;
    std::cout << "Time of function do_evaporation: " << elapsed_seconds[2].count() << std::endl;
    std::cout << "Time of function update: " << elapsed_seconds[3].count() << std::endl;
    std::cout << "Time of function display: " << elapsed_seconds[4].count() << std::endl;
}

void advance_time( const labyrinthe& land, pheromone& phen, 
                   const position_t& pos_nest, const position_t& pos_food,
                   std::vector<ant>& ants, std::size_t& cpteur )
{   
    #pragma omp parallel reduction (+: cpteur)
    {
        if(omp_get_thread_num() != 0){
            int num_thread = omp_get_thread_num() - 1;
            int size_thread = omp_get_num_threads() - 1;
            int size_block = ants.size()/size_thread;

            int start = num_thread * size_block;
            int end = start + size_block;

           // #pragma omp critical
           //std::cout << "advance " << omp_get_thread_num() << " - " <<  start << " - "  << end << std::endl;
            for ( size_t i = start; i < end; ++i ){
                ants[i].advance(phen, land, pos_food, pos_nest, cpteur);
            }
        }
        
    }
   
        
    end[1] = std::chrono::system_clock::now();
    elapsed_seconds[1] = end[1] - start[1];

    start[2] = std::chrono::system_clock::now();
    phen.do_evaporation();
    end[2] = std::chrono::system_clock::now();
    elapsed_seconds[2] = end[2] - start[2];
    
    start[3] = std::chrono::system_clock::now();
    phen.update();
    end[3] = std::chrono::system_clock::now();
    elapsed_seconds[3] = end[3] - start[3];
}

int main(int nargs, char* argv[])
{
    start[0] = std::chrono::system_clock::now();

    bool already_print = false;
    const dimension_t dims{32, 64};// Dimension du labyrinthe
    const std::size_t life = int(dims.first*dims.second);
    const int nb_ants = 2*dims.first*dims.second; // Nombre de fourmis
    const double eps = 0.75;  // Coefficient d'exploration
    const double alpha=0.97; // Coefficient de chaos
    //const double beta=0.9999; // Coefficient d'évaporation
    const double beta=0.999; // Coefficient d'évaporation
                             // 
    labyrinthe laby(dims);
    // Location du nid
    position_t pos_nest{dims.first/2,dims.second/2};
    // Location de la nourriture
    position_t pos_food{dims.first-1,dims.second-1};
                          
    
    // Définition du coefficient d'exploration de toutes les fourmis.
    ant::set_exploration_coef(eps);
    // On va créer toutes les fourmis dans le nid :
    std::vector<ant> ants;
    ants.reserve(nb_ants);
    for ( size_t i = 0; i < nb_ants; ++i )
        ants.emplace_back(pos_nest, life);
    // On crée toutes les fourmis dans la fourmilière.
    pheromone phen(laby.dimensions(), pos_food, pos_nest, alpha, beta);

    gui::context graphic_context(nargs, argv);
    gui::window& win =  graphic_context.new_window(h_scal*laby.dimensions().second,h_scal*laby.dimensions().first+266);
    display_t displayer( laby, phen, pos_nest, pos_food, ants, win );
    size_t food_quantity = 0;

    gui::event_manager manager;
    manager.on_key_event(int('q'), [] (int code) { exit(0); });
    manager.on_key_event(int('t'), [] (int code) { print_time_execution(); });
    manager.on_display([&] { displayer.display(food_quantity); win.blit(); });
    manager.on_idle([&] () { 
     
        advance_time(laby, phen, pos_nest, pos_food, ants, food_quantity);

        #pragma omp master 
        {   
           // #pragma omp critical
          //std::cout << "display " << omp_get_thread_num() << std::endl;
            start[4] = std::chrono::system_clock::now();
            displayer.display(food_quantity); 
            end[4] = std::chrono::system_clock::now();
            elapsed_seconds[4] = end[4] - start[4];
        }

        if(food_quantity >= 10000 && !already_print){
            end[0] = std::chrono::system_clock::now();
            elapsed_seconds[0] = end[0] - start[0];
            std::cout << "Time to find "<< food_quantity <<" pieces of food: " << elapsed_seconds[0].count() << std::endl;
            already_print = !already_print;
        }

        win.blit(); 
    });
    manager.loop();

    return 0;
}