/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2022 - 2024 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 *
 * Author: Simon Sticko, Uppsala University, 2021
 */

// @sect3{Include files}

// The first include files have all been treated in previous examples.

#include <deal.II/base/function.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <vector>

// The first new header contains some common level set functions.
// For example, the spherical geometry that we use here.
#include <deal.II/base/function_signed_distance.h>

// We also need 3 new headers from the NonMatching namespace.
#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

// @sect3{The HeatSolver class Template}
// We then define the main class that solves the Heat problem.

namespace Step99
{
  using namespace dealii;
  // ==================================================================
  // Analytic Solution
  // ==================================================================
  template <int dim>
  class AnalyticalSolution : public Function<dim>
  {
  public:
    double value(const Point<dim>  &point,
                 const unsigned int component = 0) const override;
  };

  template <int dim>
  double AnalyticalSolution<dim>::value(const Point<dim>  &point,
                                        const unsigned int component) const
  {
    AssertIndexRange(component, this->n_components);
    (void)component;
    const double t = this->get_time();
    return (1. - 2. / dim * (point.norm_square() - 1.))* std::exp(-t);
    //return std::pow(point[0],9) * std::pow(point[1],8) * std::exp(-t);
  }
  
  // ==================================================================
  // RHS Function
  // ==================================================================
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    double value(const Point<dim>  &point,
                 const unsigned int component = 0) const override;
  };
 
  template <int dim>
  double RightHandSide<dim>::value(const Point<dim>  &p,
                                        const unsigned int component) const
  {
    AssertIndexRange(component, this->n_components);
    (void)component;
    const double t = this->get_time();
    return - (1. - 2. / dim * (p.norm_square() - 1.))* std::exp(-t) + 4* std::exp(-t);
    //const double g = 1.0 - 2.0 / dim * (p.norm_square() - 1.0);
    //return std::exp(-t) * (4.0 - g);
    //return -std::pow(p[0], 7.0) * std::pow(p[1], 6.0) * std::exp(-t) *
    //                 (std::pow(p[0], 2.0) * std::pow(p[1], 2.0) +
    //                  72 * std::pow(p[1], 2.0) + 56 * std::pow(p[0], 2.0));
  }

  // ==================================================================
  // Boundary Values
  // ==================================================================
  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    double value(const Point<dim>  &p,
                        const unsigned int component = 0) const override;
  };

  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> &point,
                                    const unsigned int component) const
  {
    AssertIndexRange(component, this->n_components);
    (void)component;
    const double t = this->get_time();
    return (1. - 2. / dim * (point.norm_square() - 1.))* std::exp(-t);
    //return std::pow(point[0],9) * std::pow(point[1],8) * std::exp(-t);
  }


  // ==================================================================
  // Initial condition
  // ==================================================================
  template <int dim>
  class InitialCondition : public Function<dim>
  {
  public:
    double value(const Point<dim>  &p,
                        const unsigned int component = 0) const override;
  };

  template <int dim>
  double InitialCondition<dim>::value(const Point<dim> &p,
                                      const unsigned int component) const
  {
    AssertIndexRange(component, this->n_components);
    (void)component;
    return 1.0 - 2.0 / dim * (p.norm_square() - 1.0);
    //return std::pow(p[0],9) * std::pow(p[1],8);
  }

  template <int dim>
  class HeatSolver
  {
  public:
    HeatSolver();

    void run();

  private:
    void make_grid();

    void setup_discrete_level_set();

    void distribute_dofs();

    void initialize_matrices();

    void assemble_system();

    void solve(const double evaluating_time, const Vector<double> previous_solution, Vector<double> &solution_out);

    void output_results() const;

    double compute_L2_error() const;

    bool face_has_ghost_penalty(
      const typename Triangulation<dim>::active_cell_iterator &cell,
      const unsigned int face_index) const;

    const unsigned int fe_degree;
    
    
    RightHandSide<dim>      rhs_function;
    BoundaryValues<dim>   boundary_condition;
    InitialCondition<dim> initial_condition;

    Triangulation<dim> triangulation;

    // We need two separate DoFHandlers. The first manages the DoFs for the
    // discrete level set function that describes the geometry of the domain.
    const FE_Q<dim> fe_level_set;
    DoFHandler<dim> level_set_dof_handler;
    Vector<double>  level_set;

    // The second DoFHandler manages the DoFs for the solution of the Poisson
    // equation.
    hp::FECollection<dim> fe_collection;
    DoFHandler<dim>       dof_handler;
    Vector<double> solution;          // u^n
    Vector<double> old_solution;      // u^{n-1}

    NonMatching::MeshClassifier<dim> mesh_classifier;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> stiffness_matrix;
    SparseMatrix<double> system_matrix;
    Vector<double>       rhs;

    double       time;
    double       time_step;
    double       final_time;
    unsigned int timestep_number;
    
    // Theta parameter for time discretization
    // theta = 0: Forward Euler (explicit)
    // theta = 0.5: Crank-Nicolson
    // theta = 1: Backward Euler (implicit, most stable)
    const double theta;
  };



  template <int dim>
  HeatSolver<dim>::HeatSolver()
    : fe_degree(1)
    , fe_level_set(fe_degree)
    , level_set_dof_handler(triangulation)
    , dof_handler(triangulation)
    , mesh_classifier(level_set_dof_handler, level_set)
    , time(0.0)           
    , time_step(0.005)     
    , final_time(.25)     
    , timestep_number(0)
    , theta(0.0)
  {}



  // @sect3{Setting up the Background Mesh}
  // We generate a background mesh with perfectly Cartesian cells. Our domain is
  // a unit disc centered at the origin, so we need to make the background mesh
  // a bit larger than $[-1, 1]^{\text{dim}}$ to completely cover $\Omega$.
  template <int dim>
  void HeatSolver<dim>::make_grid()
  {
    std::cout << "Creating background mesh" << std::endl;
    // Triangulation<dim> triangulation_quad;
    // GridGenerator::hyper_cube(triangulation_quad, -2, 2);  
    GridGenerator::hyper_cube(triangulation, -2 , 2);
    // GridGenerator::convert_hypercube_to_simplex_mesh (triangulation_quad,
    //                                               triangulation);
    triangulation.refine_global(2);
  }



  // @sect3{Setting up the Discrete Level Set Function}
  // The discrete level set function is defined on the whole background mesh.
  // Thus, to set up the DoFHandler for the level set function, we distribute
  // DoFs over all elements in $\mathcal{T}_h$. We then set up the discrete
  // level set function by interpolating onto this finite element space.
  template <int dim>
  void HeatSolver<dim>::setup_discrete_level_set()
  {
    std::cout << "Setting up discrete level set function" << std::endl;

    level_set_dof_handler.distribute_dofs(fe_level_set);
    level_set.reinit(level_set_dof_handler.n_dofs());

    const Functions::SignedDistance::Sphere<dim> signed_distance_sphere;
    VectorTools::interpolate(level_set_dof_handler,
                             signed_distance_sphere,
                             level_set);
  }



  // @sect3{Setting up the Finite Element Space}
  // To set up the finite element space $V_\Omega^h$, we will use 2 different
  // elements: FE_Q and FE_Nothing. For better readability we define an enum for
  // the indices in the order we store them in the hp::FECollection.
  enum ActiveFEIndex
  {
    lagrange = 0,
    nothing  = 1
  };

  // We then use the MeshClassifier to check LocationToLevelSet for each cell in
  // the mesh and tell the DoFHandler to use FE_Q on elements that are inside or
  // intersected, and FE_Nothing on the elements that are outside.
  template <int dim>
  void HeatSolver<dim>::distribute_dofs()
  {
    std::cout << "Distributing degrees of freedom" << std::endl;

    fe_collection.push_back(FE_Q<dim>(fe_degree));
    fe_collection.push_back(FE_Nothing<dim>());

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        const NonMatching::LocationToLevelSet cell_location =
          mesh_classifier.location_to_level_set(cell);

        if (cell_location == NonMatching::LocationToLevelSet::outside)
          cell->set_active_fe_index(ActiveFEIndex::nothing);
        else
          cell->set_active_fe_index(ActiveFEIndex::lagrange);
      }

    dof_handler.distribute_dofs(fe_collection);
  }

  template <int dim>
  void HeatSolver<dim>::initialize_matrices()
  {
    std::cout << "Initializing matrices" << std::endl;

    const auto face_has_flux_coupling = [&](const auto        &cell,
                                            const unsigned int face_index) {
      return this->face_has_ghost_penalty(cell, face_index);
    };

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

    const unsigned int           n_components = fe_collection.n_components();
    Table<2, DoFTools::Coupling> cell_coupling(n_components, n_components);
    Table<2, DoFTools::Coupling> face_coupling(n_components, n_components);
    cell_coupling[0][0] = DoFTools::always;
    face_coupling[0][0] = DoFTools::always;

    const AffineConstraints<double> constraints;
    const bool                      keep_constrained_dofs = true;

    DoFTools::make_flux_sparsity_pattern(dof_handler,
                                         dsp,
                                         constraints,
                                         keep_constrained_dofs,
                                         cell_coupling,
                                         face_coupling,
                                         numbers::invalid_subdomain_id,
                                         face_has_flux_coupling);
    sparsity_pattern.copy_from(dsp);

    mass_matrix.reinit(sparsity_pattern);
    stiffness_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);    
    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    rhs.reinit(dof_handler.n_dofs());
  }



  // The following function describes which faces are part of the set
  // $\mathcal{F}_h$. That is, it returns true if the face of the incoming cell
  // belongs to the set $\mathcal{F}_h$.
  template <int dim>
  bool HeatSolver<dim>::face_has_ghost_penalty(
    const typename Triangulation<dim>::active_cell_iterator &cell,
    const unsigned int                                       face_index) const
  {
    if (cell->at_boundary(face_index))
      return false;

    const NonMatching::LocationToLevelSet cell_location =
      mesh_classifier.location_to_level_set(cell);

    const NonMatching::LocationToLevelSet neighbor_location =
      mesh_classifier.location_to_level_set(cell->neighbor(face_index));

    if (cell_location == NonMatching::LocationToLevelSet::intersected &&
        neighbor_location != NonMatching::LocationToLevelSet::outside)
      return true;

    if (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
        cell_location != NonMatching::LocationToLevelSet::outside)
      return true;

    return false;
  }



  // @sect3{Assembling the System}
  template <int dim>
  void HeatSolver<dim>::assemble_system()
  {
    std::cout << "Assembling" << std::endl;

    const unsigned int n_dofs_per_cell = fe_collection[0].dofs_per_cell;
    FullMatrix<double> local_mass(n_dofs_per_cell, n_dofs_per_cell);
    FullMatrix<double> local_stiffness(n_dofs_per_cell, n_dofs_per_cell);

    // The below local_rhs will be assembled later on because now it will depend upon time value too while the LHS system matrix is independent of time. Consequently
    // all the rhs assembly is deleted
    // Vector<double>     local_rhs(n_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

    const double ghost_parameter_1   = 0.75;
    const double ghost_parameter_2   = 1.5;
    const double nitsche_parameter = 5 * (fe_degree) * fe_degree;

    // Since the ghost penalty is similar to a DG flux term, the simplest way to
    // assemble it is to use an FEInterfaceValues object.
    const QGauss<dim - 1>  face_quadrature(fe_degree + 1);
    FEInterfaceValues<dim> fe_interface_values(fe_collection[0],
                                               face_quadrature,
                                               update_gradients |
                                                 update_JxW_values |
                                                 update_normal_vectors);

    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);

    // As we iterate over the cells, we don't need to do anything on the cells
    // that have FE_Nothing elements. To disregard them we use an iterator
    // filter.
    for (const auto &cell :
         dof_handler.active_cell_iterators() |
           IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::lagrange))
      {
        local_mass = 0;
        local_stiffness = 0;

        const double cell_side_length = cell->minimum_vertex_distance();

        non_matching_fe_values.reinit(cell);

        const std::optional<FEValues<dim>> &inside_fe_values =
          non_matching_fe_values.get_inside_fe_values();

        if (inside_fe_values)
          for (const unsigned int q :
               inside_fe_values->quadrature_point_indices())
            {
              // const Point<dim> &point = inside_fe_values->quadrature_point(q);
              for (const unsigned int i : inside_fe_values->dof_indices())
                {
                  for (const unsigned int j : inside_fe_values->dof_indices())
                    {
                      local_mass(i, j) +=
                        inside_fe_values->shape_value(i, q) *
                        inside_fe_values->shape_value(j, q) *
                        inside_fe_values->JxW(q);

                      local_stiffness(i, j) +=
                        inside_fe_values->shape_grad(i, q) *
                        inside_fe_values->shape_grad(j, q) *
                        inside_fe_values->JxW(q);
                    }
                  // local_rhs(i) += rhs_function.value(point) *
                  //                   inside_fe_values->shape_value(i, q) *
                  //                   inside_fe_values->JxW(q);
                }
            }

        const std::optional<NonMatching::FEImmersedSurfaceValues<dim>>
          &surface_fe_values = non_matching_fe_values.get_surface_fe_values();

        if (surface_fe_values)
          {
            for (const unsigned int q :
                 surface_fe_values->quadrature_point_indices())
              {
                // const Point<dim> &point =
                //   surface_fe_values->quadrature_point(q);
                const Tensor<1, dim> &normal =
                  surface_fe_values->normal_vector(q);
                for (const unsigned int i : surface_fe_values->dof_indices())
                  {
                    for (const unsigned int j :
                         surface_fe_values->dof_indices())
                      {
                        local_stiffness(i, j) +=
                          (-normal * surface_fe_values->shape_grad(i, q) *
                             surface_fe_values->shape_value(j, q) +
                           -normal * surface_fe_values->shape_grad(j, q) *
                             surface_fe_values->shape_value(i, q) +
                           nitsche_parameter / cell_side_length *
                             surface_fe_values->shape_value(i, q) *
                             surface_fe_values->shape_value(j, q)) *
                          surface_fe_values->JxW(q);
                      }
                    // local_rhs(i) +=
                    //   boundary_condition.value(point) *
                    //   (nitsche_parameter / cell_side_length *
                    //      surface_fe_values->shape_value(i, q) -
                    //    normal * surface_fe_values->shape_grad(i, q)) *
                    //   surface_fe_values->JxW(q);
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);

        mass_matrix.add(local_dof_indices, local_mass);  
        stiffness_matrix.add(local_dof_indices, local_stiffness);

        for (const unsigned int f : cell->face_indices())
          if (face_has_ghost_penalty(cell, f))
            {
              const unsigned int invalid_subface =
                numbers::invalid_unsigned_int;

              fe_interface_values.reinit(cell,
                                         f,
                                         invalid_subface,
                                         cell->neighbor(f),
                                         cell->neighbor_of_neighbor(f),
                                         invalid_subface);

              const unsigned int n_interface_dofs =
                fe_interface_values.n_current_interface_dofs();
              FullMatrix<double> local_mass_stabilization(n_interface_dofs,
                                                     n_interface_dofs);
              FullMatrix<double> local_stabilization(n_interface_dofs,
                                                     n_interface_dofs);
              for (unsigned int q = 0;
                   q < fe_interface_values.n_quadrature_points;
                   ++q)
                {
                  const Tensor<1, dim> normal =
                    fe_interface_values.normal_vector(q);
                  for (unsigned int i = 0; i < n_interface_dofs; ++i)
                    for (unsigned int j = 0; j < n_interface_dofs; ++j)
                      {
                        local_mass_stabilization(i, j) +=
                          .5 * ghost_parameter_1 * (0.33) * std::pow(cell_side_length,3) * normal *
                          fe_interface_values.jump_in_shape_gradients(i, q) *
                          normal *
                          fe_interface_values.jump_in_shape_gradients(j, q) *
                          fe_interface_values.JxW(q);                        
                        local_stabilization(i, j) +=
                          .5 * ghost_parameter_2 * (0.33) *  cell_side_length * normal *
                          fe_interface_values.jump_in_shape_gradients(i, q) *
                          normal *
                          fe_interface_values.jump_in_shape_gradients(j, q) *
                          fe_interface_values.JxW(q);
                      }
                }

              const std::vector<types::global_dof_index>
                local_interface_dof_indices =
                  fe_interface_values.get_interface_dof_indices();

              mass_matrix.add(local_interface_dof_indices,
                                   local_mass_stabilization);
              stiffness_matrix.add(local_interface_dof_indices,
                                   local_stabilization);
            }
      }

    system_matrix.copy_from(mass_matrix);
    // system_matrix.add(theta * time_step, stiffness_matrix);
  }


  // @sect3{Solving the System}
  template <int dim>
  void HeatSolver<dim>::solve(const double evaluating_time, const Vector<double> previous_solution, Vector<double> &solution_out)
  {

    rhs = 0;

    if (theta < 1.0)
      {
        Vector<double> tmp(solution.size());
        stiffness_matrix.vmult(tmp, previous_solution);
        rhs.add(-1.0, tmp);
      }

    const unsigned int n_dofs_per_cell = fe_collection[0].dofs_per_cell;
    Vector<double> local_rhs(n_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

    const double nitsche_parameter = 5 * (fe_degree) * fe_degree;

    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_JxW_values | 
                                 update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);

    for (const auto &cell :
         dof_handler.active_cell_iterators() |
           IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::lagrange))
      {
        local_rhs = 0;
        const double cell_side_length = cell->minimum_vertex_distance();

        non_matching_fe_values.reinit(cell);

        // ============================================================
        // VOLUME SOURCE TERM: ∫ f φᵢ dx
        // ============================================================
        const std::optional<FEValues<dim>> &inside_fe_values =
          non_matching_fe_values.get_inside_fe_values();

        if (inside_fe_values)
          {
            for (const unsigned int q :
                 inside_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = inside_fe_values->quadrature_point(q);
                
                // Evaluate f at θ*t^n + (1-θ)*t^{n-1}
                rhs_function.set_time(evaluating_time);
                const double f_value = rhs_function.value(point);

                for (const unsigned int i : inside_fe_values->dof_indices())
                  {
                    local_rhs(i) += f_value *
                                    inside_fe_values->shape_value(i, q) *
                                    inside_fe_values->JxW(q);
                  }
              }
          }

        // ============================================================
        // BOUNDARY TERMS: Nitsche RHS
        // ============================================================
        const std::optional<NonMatching::FEImmersedSurfaceValues<dim>>
          &surface_fe_values = non_matching_fe_values.get_surface_fe_values();

        if (surface_fe_values)
          {
            for (const unsigned int q :
                 surface_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point =
                  surface_fe_values->quadrature_point(q);
                const Tensor<1, dim> &normal =
                  surface_fe_values->normal_vector(q);

                // Evaluate g at θ*t^n + (1-θ)*t^{n-1}
                boundary_condition.set_time(evaluating_time);
                const double g_value = boundary_condition.value(point);

                for (const unsigned int i : surface_fe_values->dof_indices())
                  {
                    local_rhs(i) +=
                      g_value *
                      (nitsche_parameter / cell_side_length *
                         surface_fe_values->shape_value(i, q) -
                       normal * surface_fe_values->shape_grad(i, q)) *
                      surface_fe_values->JxW(q);
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);
        rhs.add(local_dof_indices, local_rhs);
      }

    std::cout << "Solving system" << std::endl;

    const unsigned int max_iterations = solution.size();
    ReductionControl      solver_control(max_iterations,1e-20,1e-10);
    SolverCG<>         solver(solver_control);
    solver.solve(system_matrix, solution_out, rhs, PreconditionIdentity());
  }



  // @sect3{Data Output}
  // Since both DoFHandler instances use the same triangulation, we can add both
  // the level set function and the solution to the same vtu-file. Further, we
  // do not want to output the cells that have LocationToLevelSet value outside.
  // To disregard them, we write a small lambda function and use the
  // set_cell_selection function of the DataOut class.
  template <int dim>
  void HeatSolver<dim>::output_results() const
  {
    std::cout << "Writing vtu file" << std::endl;

    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, solution, "solution");
    data_out.add_data_vector(level_set_dof_handler, level_set, "level_set");

    Vector<double> analytical_solution;
    analytical_solution.reinit(solution);

    AnalyticalSolution<dim> analytical_solution_fu;
    analytical_solution_fu.set_time(time);

    VectorTools::interpolate(dof_handler,
                             analytical_solution_fu,
                             analytical_solution);

    data_out.add_data_vector(dof_handler, analytical_solution, "analytical");

    data_out.set_cell_selection(
      [this](const typename Triangulation<dim>::cell_iterator &cell) {
        return cell->is_active() &&
               mesh_classifier.location_to_level_set(cell) !=
                 NonMatching::LocationToLevelSet::outside;
      });

    data_out.build_patches();
    std::ofstream output("step-99.vtu");
    data_out.write_vtu(output);
  }

  template <int dim>
  double HeatSolver<dim>::compute_L2_error() const
  {
    std::cout << "Computing L2 error" << std::endl;

    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside =
      update_values | update_JxW_values | update_quadrature_points;

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);

    // We then iterate iterate over the cells that have LocationToLevelSetValue
    // value inside or intersected again. For each quadrature point, we compute
    // the pointwise error and use this to compute the integral.
    AnalyticalSolution<dim> analytical_solution;
    analytical_solution.set_time(time);
    double                  error_L2_squared = 0;

    for (const auto &cell :
         dof_handler.active_cell_iterators() |
           IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::lagrange))
      {
        non_matching_fe_values.reinit(cell);

        const std::optional<FEValues<dim>> &fe_values =
          non_matching_fe_values.get_inside_fe_values();

        if (fe_values)
          {
            std::vector<double> solution_values(fe_values->n_quadrature_points);
            fe_values->get_function_values(solution, solution_values);

            for (const unsigned int q : fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = fe_values->quadrature_point(q);
                const double      error_at_point =
                  solution_values.at(q) - analytical_solution.value(point);
                error_L2_squared +=
                  Utilities::fixed_power<2>(error_at_point) * fe_values->JxW(q);
              }
          }
      }

    return std::sqrt(error_L2_squared);
  }



  // @sect3{A Convergence Study}
  // Finally, we do a convergence study to check that the $L^2$-error decreases
  // with the expected rate. We refine the background mesh a few times. In each
  // refinement cycle, we solve the problem, compute the error, and add the
  // $L^2$-error and the mesh size to a ConvergenceTable.
  template <int dim>
  void HeatSolver<dim>::run()
  {
    ConvergenceTable   convergence_table;
    const unsigned int n_refinements = 3;

    make_grid();
    // std::vector<double> prev_error;
    double prev_error = 0.0;
    double prev_h = 0.0;
    for (unsigned int cycle = 0; cycle <= n_refinements; cycle++)
      {
        std::cout << "Refinement cycle " << cycle << std::endl;
        triangulation.refine_global(1);
        time = 0.0;
        timestep_number = 0;
        const double cell_side_length =
          triangulation.begin_active()->minimum_vertex_distance();
        time_step = (0.05)*std::pow(cell_side_length,2);
        setup_discrete_level_set();
        std::cout << "Classifying cells" << std::endl;
        mesh_classifier.reclassify();
        distribute_dofs();
        initialize_matrices();
        assemble_system();
        VectorTools::interpolate(dof_handler,
                               initial_condition,
                               old_solution);

        double error_L2;               
        
        while(time<final_time-1e-6)
        {
          Vector<double> sol_k1(dof_handler.n_dofs());
          Vector<double> sol_k2(dof_handler.n_dofs());
          Vector<double> sol_k3(dof_handler.n_dofs());
          Vector<double> sol_k4(dof_handler.n_dofs());
          Vector<double> tmp(dof_handler.n_dofs());

          // k1
          solve(time, old_solution, sol_k1);

          // k2
          tmp = old_solution;
          tmp.add(time_step / 2.0, sol_k1);
          solve(time + time_step / 2.0, tmp, sol_k2);

          // k3
          tmp = old_solution;
          tmp.add(time_step / 2.0, sol_k2);
          solve(time + time_step / 2.0, tmp, sol_k3);

          // k4
          tmp = old_solution;
          tmp.add(time_step, sol_k3);
          solve(time + time_step, tmp, sol_k4);

          // Final combination
          solution = old_solution;

          solution.add(time_step / 6.0, sol_k1);
          solution.add(time_step / 3.0, sol_k2);   // 2/6 = 1/3
          solution.add(time_step / 3.0, sol_k3);
          solution.add(time_step / 6.0, sol_k4);
          error_L2 = compute_L2_error();
          time += time_step;
          timestep_number += 1;
          std::cout<<time<<"    "<<timestep_number<<std::endl;
          old_solution = solution;
        }

        output_results();

        convergence_table.add_value("Cycle", cycle);
        convergence_table.add_value("Mesh size", cell_side_length);
        convergence_table.add_value("Time Step", time_step);
        convergence_table.add_value("Time Step Number", timestep_number);
        convergence_table.add_value("L2-Error", error_L2);

        if (cycle > 0)
        {
          const double rate =
            std::log(error_L2 / prev_error) /
            std::log(cell_side_length / prev_h);

          convergence_table.add_value("Rate", rate);
        }
        else
        {
          convergence_table.add_value("Rate", 0.0);
        }

        prev_error = error_L2;
        prev_h = cell_side_length;

        std::cout << std::endl;
        convergence_table.write_text(std::cout);
        std::cout << std::endl;

        

      }
  }

} // namespace Step99



// @sect3{The main() function}
int main()
{
  const int dim = 2;

  Step99::HeatSolver<dim> heat_solver;
  heat_solver.run();
}
