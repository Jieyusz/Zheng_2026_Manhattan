"""
Generate the endotaxis-model figure data for the algorithm figure (fig:algo) and the
Mask-D bottleneck-transition panels (fig:maskd / fig:maskd_supp).

Runs the endotaxis navigation model along the example animals' first-journey corridor
sequences (Mask A linear track and Mask D), and builds the outskirt-removed Mask-D
corridor-transition matrices / adjacency used as the schematic index space. Consumes the
example-session caches written earlier in the pipeline, so it runs last (see
``batch_generate_figure_data.py`` phase 4).

Saved keys
----------
"Wildtype A example corridor seq"          : (T,) graph-distance-ordered corridor sequence, example Mask-A first journey.
"Wildtype A example learned adjacency Out" : list of learned corridor adjacency matrices over learning (Mask A, Out goal).
"Wildtype A example learned signal Out"    : log goal-signal over learning (Mask A, Out goal).
"Wildtype D endotaxis corridor order"      : (corridor_index_order, order_operator) for the outskirt-removed 18-corridor display space.
"Wildtype D corridor transition matrices"  : (n_sessions, size, 18, 18) directional P(start->end) in display order.
"Wildtype D {biclique 1,2} offpath transitions" / "... offpath choice ratios"
                                           : off-shortest-path within-biclique transitions (raw counts / normalized choice ratios).
"Wildtype D shortest path ordered"         : shortest-path corridors in display order.
"Wildtype D example corridor seq"          : example Mask-D first-journey corridor sequence in display order.
"Wildtype D corridor adjacency"            : (18, 18) static reduced corridor graph in display order.
"Wildtype D example learned signal {Out,Home}" / "... learned adjacency {Out,Home}"
                                           : endotaxis learned signal / adjacency over learning, per goal (Mask D).
See docs/data_contracts.md §12 for shapes/units and the consuming plot scripts.

Run (m_maze env, from scripts/, repo on PYTHONPATH):
    python gen_endotaxis.py --overwrite
Also picked up automatically by ``batch_generate_figure_data.py`` (final phase).
"""
import manhattan_maze as mm
from manhattan_maze import utils, endotaxis
from manhattan_maze import mask_d
import argparse
import numpy as np
import config


def main():
    parser = argparse.ArgumentParser(description="Generate figure data")
    parser.add_argument("-ow", "--overwrite", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    overwrite = args.overwrite

    ## shared paths and DataLoader configuration (see scripts/config.py)
    save_dir = config.SAVE_DIR
    data = mm.DataLoader(config.DATA_DIR, **config.DATALOADER_KWARGS)
    mdf = data.metadata
    bl6j_mdf = mdf[mdf.Genotype=="BL6J"]
    figure_data_dict = utils.load_all_figure_data()

    # === Endotaxis model on Mask A (example animal) ===
    # Run the endotaxis learning model along the example animal's first-journey
    # corridor sequence and save the learned adjacency matrix and goal signal over
    # learning, used to illustrate the model's solution on the simple linear track.
    print("Learning Mask A")

    example_id = config.MASK_A_EXAMPLE_ID
    # The Mask-A example animals are chosen by a seeded np.random.choice in
    # gen_wildtype_two_day_data.py, so which animal this is cannot be recomputed here --
    # but it is recorded in that script's exported manifest. Read the identity from the
    # manifest and reload the live Session from the DataLoader, rather than unpickling a
    # Session out of the figure cache (R8).
    a_manifest = figure_data_dict["Mask A example manifest"]
    a_row = a_manifest[a_manifest.example == example_id].iloc[0]
    a_session = data[a_row.animal_name][int(a_row.session_idx)]
    a_slice = a_session.slice_by_traverse_idx(None, 1)
    a_corridor_seq = [b.get_corridor_distance_seq(goal_corridor=a_session.mask.home_corridor) for b in a_slice] # order based on graph distance
    # flatten nested list
    a_corridor_seq = np.concatenate(a_corridor_seq)
    utils.save_modular_data("Wildtype A example corridor seq", a_corridor_seq, save_dir, overwrite=overwrite)
    ga, th, endotaxis_alpha, de = config.ENDOTAXIS_LEARNING_PARAMETERS # gain, threshold, endotaxis_alpha (learning rate), decay (C2)
    a_adjcency_matrix = np.eye(10, k=-1) + np.eye(10, k=1) # adjacency matrix only for corridors
    Nsa, Msa, Gsa = endotaxis.Learn_Mouse_tr(a_adjcency_matrix, a_corridor_seq, ga, th, endotaxis_alpha, de, bi=True,
                                              goal=config.ENDOTAXIS_MASK_A_GOAL_CORRIDOR)
    utils.save_modular_data("Wildtype A example learned adjacency Out", Msa, save_dir, overwrite=overwrite)

    Ssa = [g[0] @ endotaxis.map_lin(M, ga) for g, M in zip(Gsa, Msa)]  # compute all the goal signals
    Ssa = np.minimum(Ssa, 1.0)  # saturate at 1.0
    Ssa = np.log(Ssa+0.1)
    utils.save_modular_data("Wildtype A example learned signal Out", Ssa, save_dir, overwrite=overwrite)

    # === Endotaxis model on Mask D (example animal, Out and Home goals) ===
    # Order corridors for plotting, then run the endotaxis model on the example
    # animal's first-journey corridor sequence for both the Out and Home goals;
    # save the corridor ordering, shortest path, and the learned signal/adjacency
    # matrices (reordered for the heatmap figure).
    print("Learning Mask D")
    d_example_id = config.MASK_D_EXAMPLE_ID
    # use the example session from main figure (plot_d_motif). The Mask-D examples are the
    # last three wildtype Day-1 sessions -- a deterministic selection -- so this
    # reproduces gen_wildtype_d_data.py exactly without reading its cache (R8).
    wildtype_d_sessions = utils.get_wildtype_d_sessions(data, mdf)
    example_session = wildtype_d_sessions[-3:][d_example_id]
    example_slice = example_session.slice_by_traverse_idx(None, 1) # get the first two journeys
    corridor_seq = endotaxis.extract_corridor_seq(example_slice)

    maskd_special_params = mask_d.MaskDSpec() # use the corridor to order
    corridor_index_order = endotaxis.remove_out_skirt(maskd_special_params.plot_corridor_order)
    order_operator = np.argsort(corridor_index_order) # use this order operator to sort the results
    utils.save_modular_data("Wildtype D endotaxis corridor order", (corridor_index_order, order_operator), save_dir, overwrite=overwrite)

    # Per-session/per-slice corridor transition matrices for the bottleneck-transition
    # figure, restricted+reordered to the outskirt-removed display corridor order so they
    # share the endotaxis schematic's index space (consumed by plot_d_full / plot_d_supp).
    d_transition_matrices = utils.get_d_transition_matrices(
        wildtype_d_sessions, size=50, corridor_order=maskd_special_params.plot_corridor_order)
    utils.save_modular_data("Wildtype D corridor transition matrices", d_transition_matrices, save_dir, overwrite=overwrite)
    # Off-shortest-path within-biclique transitions (3x3x2 per biclique): all direction
    # transitions between the off-path corridors of each biclique's two groups, keyed by raw
    # (start, end) corridor pairs (consumed downstream as a quantitative off-path companion).
    for biclique_name, biclique_groups in [("biclique 1", maskd_special_params.biclique_1_groups),
                                           ("biclique 2", maskd_special_params.biclique_2_groups)]:
        offpath_transitions = utils.select_biclique_offpath_transitions(
            d_transition_matrices, biclique_groups,
            maskd_special_params.shortest_path_corridor_indices, maskd_special_params.plot_corridor_order)
        utils.save_modular_data(f"Wildtype D {biclique_name} offpath transitions",
                                offpath_transitions, save_dir, overwrite=overwrite)
        # Choice ratios: same transitions normalized over the 4 possible choices from each
        # start node (its opposite biclique group, including the shortest-path arm).
        offpath_choice_ratios = utils.select_biclique_offpath_transitions(
            d_transition_matrices, biclique_groups,
            maskd_special_params.shortest_path_corridor_indices, maskd_special_params.plot_corridor_order,
            normalize=True)
        utils.save_modular_data(f"Wildtype D {biclique_name} offpath choice ratios",
                                offpath_choice_ratios, save_dir, overwrite=overwrite)
    shortest_path = endotaxis.remove_out_skirt(maskd_special_params.shortest_path_corridor_indices)
    utils.save_modular_data("Wildtype D shortest path ordered", order_operator[shortest_path], save_dir, overwrite=overwrite)
    utils.save_modular_data("Wildtype D example corridor seq", order_operator[corridor_seq], save_dir, overwrite=overwrite)
    mask_d_wo_skirt = data.masks["D"].remove_outskirts()
    # Static reduced corridor adjacency in display order — backdrop graph for the
    # bottleneck-transition schematic (plot_d_full / plot_d_supp).
    corridor_adjacency = mask_d_wo_skirt.corridors_adj_mat[np.ix_(corridor_index_order, corridor_index_order)]
    utils.save_modular_data("Wildtype D corridor adjacency", corridor_adjacency, save_dir, overwrite=overwrite)
    home_corr = config.ENDOTAXIS_MASK_D_HOME_CORRIDOR # when outskirt is removed
    goal_corr = config.ENDOTAXIS_MASK_D_OUT_CORRIDOR
    # learn
    for goal_name, learning_goals in zip(["Out", "Home"], [[home_corr,goal_corr],[goal_corr,home_corr]]):
        start, goal = learning_goals
        Ns, Ms, Gs = endotaxis.Learn_Mouse_tr(mask_d_wo_skirt.corridors_adj_mat, corridor_seq, ga, th, endotaxis_alpha, de, bi=True,
                                              goal=goal)
        Ss=[g[0]@endotaxis.map_lin(M,ga) for g,M in zip(Gs,Ms)] # compute all the goal signals
        Ss=np.minimum(Ss,1.0) # saturate at 1.0
        Ss = Ss[:, corridor_index_order]
        Ss = np.log(Ss+0.1)
        utils.save_modular_data(f"Wildtype D example learned signal {goal_name}", Ss, save_dir, overwrite=overwrite)
        # also save adjacency matrix
        Msd = [M[np.ix_(corridor_index_order, corridor_index_order)] for M in Ms]
        utils.save_modular_data(f"Wildtype D example learned adjacency {goal_name}", Msd, save_dir, overwrite=overwrite)


if __name__ == "__main__":
    main()
