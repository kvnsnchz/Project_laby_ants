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
    //#pragma omp parallel for schedule(dynamic, 64) reduction(+:cpteur)
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
    int nbp, rank;
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
    MPI_Group world_group, group_0_1, group_1_n;
    MPI_Comm comm_0_1, comm_1_n;

    MPI_Init(&nargs, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nbp);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int ranks_0_1[2] = {0, 1};
    int ranks_1_n[nbp - 1];
    int nb_ants_proc = nb_ants / (nbp - 1);
    int nb_ants_total = nb_ants_proc*(nbp - 1);
    const int buffer_size = 1 + 2*nb_ants_total + laby.dimensions().first*laby.dimensions().second;
   
    for(int i = 1; i < nbp; i++){
        ranks_1_n[i - 1] = i; 
    }

    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    if(rank == 1 || rank == 0){
        MPI_Group_incl(world_group, 2, ranks_0_1, &group_0_1);
        MPI_Comm_create_group(MPI_COMM_WORLD, group_0_1, 0, &comm_0_1);
    }
    if(rank >= 1){
        MPI_Group_incl(world_group, nbp - 1, ranks_1_n, &group_1_n);
        MPI_Comm_create_group(MPI_COMM_WORLD, group_1_n, 0, &comm_1_n);   
    }

    if(rank >= 1){
        const double eps = 0.75;  // Coefficient d'exploration
        size_t food_quantity = 0;
        int new_rank;
        // Définition du coefficient d'exploration de toutes les fourmis.
        ant::set_exploration_coef(eps);
        // On va créer toutes les fourmis dans le nid :
        std::vector<ant> ants;
        ants.reserve(nb_ants_proc);
        for ( size_t i = 0; i < nb_ants_proc; ++i )
            ants.emplace_back(pos_nest, life);
        // On crée toutes les fourmis dans la fourmilière.

        pheromone phen(laby.dimensions(), pos_food, pos_nest, alpha, beta);
       
        while(true){
            std::vector<double> ants_buffer, ants_buffer_recv, pher_buffer_recv;
            int food_quantity_buffer;
           
            for(size_t i = 0; i < nb_ants_proc; ++i){
                position_t ant_pos = ants[i].get_position();
                ants_buffer.emplace_back((double)ant_pos.first);
                ants_buffer.emplace_back((double)ant_pos.second);
            }

            MPI_Request request;
            MPI_Status status;
            
            if(rank == 1){
                std::vector<double>(2*nb_ants_total).swap(ants_buffer_recv);
            }

            std::vector<double>(laby.dimensions().first * laby.dimensions().second).swap(pher_buffer_recv);
            MPI_Reduce(&food_quantity, &food_quantity_buffer, 1, MPI_INT, MPI_SUM, 0, comm_1_n);
            MPI_Gather(ants_buffer.data(), ants_buffer.size(), MPI_DOUBLE, ants_buffer_recv.data(), ants_buffer.size(), MPI_DOUBLE, 0, comm_1_n);
            MPI_Allreduce(phen.get_m_map_of_pheromone().data(), pher_buffer_recv.data(),laby.dimensions().first * laby.dimensions().second, MPI_DOUBLE, MPI_MAX, comm_1_n);
            phen.copy(pher_buffer_recv);
            
            if(rank == 1){
                std::vector<double> buffer;
                buffer.emplace_back((double)food_quantity_buffer);
                buffer.insert(buffer.end(), ants_buffer_recv.begin(), ants_buffer_recv.end());
                buffer.insert(buffer.end(), pher_buffer_recv.begin(), pher_buffer_recv.end());
                
                MPI_Isend(buffer.data(), buffer.size(), MPI_DOUBLE, 0, 101, comm_0_1, &request);        
                advance_time(laby, phen, pos_nest, pos_food, ants, food_quantity);
                MPI_Wait(&request, &status);
            }
            else{
                advance_time(laby, phen, pos_nest, pos_food, ants, food_quantity);
            }
            
        }
        
    }

    if(rank == 0){
        start[0] = std::chrono::system_clock::now();
        std::vector<ant> ants;
        std::vector<double> buffer_rec(buffer_size);
        ants.reserve(nb_ants_total);
        size_t food_quantity = 0;
        size_t ants_start = 1;
        size_t pher_start = nb_ants_total*2 + 1;
        pheromone phen(laby.dimensions(), pos_food, pos_nest, alpha, beta);
        
        MPI_Status status;
        
        MPI_Recv(buffer_rec.data(), buffer_rec.size(), MPI_DOUBLE, 1, 101, comm_0_1, &status);

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

            MPI_Recv(buffer_rec.data(), buffer_rec.size(), MPI_DOUBLE, 1, 101, comm_0_1, &status);
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