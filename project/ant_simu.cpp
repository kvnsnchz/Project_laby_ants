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
#include <mpi.h>

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
    start[1] = std::chrono::system_clock::now();
    #pragma omp parallel for schedule(dynamic, 64) reduction(+:cpteur)
    for ( size_t i = 0; i < ants.size(); ++i )
        ants[i].advance(phen, land, pos_food, pos_nest, cpteur);
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
    bool already_print = false;
    int nbp, rank, provided;
    const dimension_t dims{32,64};// Dimension du labyrinthe
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
    const int buffer_size = 1 + 2*nb_ants + laby.dimensions().first*laby.dimensions().second;

    MPI_Init_thread(&nargs, &argv, MPI_THREAD_SERIALIZED, &provided);
    if(provided < MPI_THREAD_SERIALIZED){
        std::cout << "Error" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &nbp);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 1){
        const double eps = 0.75;  // Coefficient d'exploration
        size_t food_quantity = 0;

        // Définition du coefficient d'exploration de toutes les fourmis.
        ant::set_exploration_coef(eps);
        // On va créer toutes les fourmis dans le nid :
        std::vector<ant> ants;
        ants.reserve(nb_ants);
        for ( size_t i = 0; i < nb_ants; ++i )
            ants.emplace_back(pos_nest, life);
        // On crée toutes les fourmis dans la fourmilière.
        pheromone phen(laby.dimensions(), pos_food, pos_nest, alpha, beta);
        while(true){
            std::vector<double> buffer;
            
            buffer.emplace_back((double)food_quantity);
            
            for(size_t i = 0; i < nb_ants; ++i){
                position_t ant_pos = ants[i].get_position();
                buffer.emplace_back((double)ant_pos.first);
                buffer.emplace_back((double)ant_pos.second);
            }

           for ( std::size_t i = 0; i < laby.dimensions( ).first; ++i )
                for ( std::size_t j = 0; j < laby.dimensions( ).second; ++j ) {
                    buffer.emplace_back((double)phen( i, j ));
                }
            
            MPI_Request request;
            MPI_Status status;
            MPI_Isend(buffer.data(), buffer.size(), MPI_DOUBLE, 0, 101, MPI_COMM_WORLD, &request);        
            advance_time(laby, phen, pos_nest, pos_food, ants, food_quantity);
            MPI_Wait(&request, &status);
        }
        
    }

    if(rank == 0){
        start[0] = std::chrono::system_clock::now();

        std::vector<ant> ants;
        ants.reserve(nb_ants);
        size_t food_quantity = 0;
        size_t ants_start = 1;
        size_t pher_start = nb_ants*2 + 1;

        pheromone phen(laby.dimensions(), pos_food, pos_nest, alpha, beta);
        
        MPI_Status status;
        std::vector<double> buffer_rec(buffer_size);
        
        MPI_Recv(buffer_rec.data(), buffer_rec.size(), MPI_DOUBLE, 1, 101, MPI_COMM_WORLD, &status);
        food_quantity = buffer_rec[0];
        for(size_t i = ants_start; i < pher_start; i += 2){
            ants.emplace_back(position_t(buffer_rec[i], buffer_rec[i+1]), life);
        }
        phen.copy(std::vector<double>(buffer_rec.begin() + pher_start, buffer_rec.end()));

        gui::context graphic_context(nargs, argv);
        gui::window& win =  graphic_context.new_window(h_scal*laby.dimensions().second,h_scal*laby.dimensions().first+266);
    
        display_t displayer( laby, phen, pos_nest, pos_food, ants, win );
        
        gui::event_manager manager;
        manager.on_key_event(int('q'), [] (int code) { MPI_Abort(MPI_COMM_WORLD, MPI_SUCCESS);});
        manager.on_key_event(int('t'), [] (int code) { print_time_execution(); });
        manager.on_display([&] { displayer.display(food_quantity); win.blit(); });
        manager.on_idle([&] () { 
            //advance_time(laby, phen, pos_nest, pos_food, ants, food_quantity);

            start[4] = std::chrono::system_clock::now();
            displayer.display(food_quantity); 
            end[4] = std::chrono::system_clock::now();
            elapsed_seconds[4] = end[4] - start[4];
            if(food_quantity >= 10000 && !already_print){
                end[0] = std::chrono::system_clock::now();
                elapsed_seconds[0] = end[0] - start[0];
                std::cout << "Time to find " << food_quantity << " pieces of food: " << elapsed_seconds[0].count() << std::endl;
                already_print = !already_print;
            }

            win.blit();

            MPI_Recv(buffer_rec.data(), buffer_rec.size(), MPI_DOUBLE, 1, 101, MPI_COMM_WORLD, &status);
            food_quantity = buffer_rec[0];
            for(size_t i = ants_start, j = 0; i < pher_start; i += 2, ++j){
                ants[j].set_position(position_t(buffer_rec[i], buffer_rec[i+1]));
            }
            phen.copy(std::vector<double>(buffer_rec.begin() + pher_start, buffer_rec.end()));
        });
        manager.loop();
    }

    MPI_Finalize();
    return 0;
}