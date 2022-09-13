from maggma.builders.map_builder import MapBuilder
from maggma.stores import MongoStore
from pymatgen.ext.matproj import MPRester  # doesn't work for emmet
from emmet.core.mobility.migrationgraph import MigrationGraphDoc
import numpy as np
from pymatgen.core import Structure


class RankElecBuilder(MapBuilder):
    def __init__(
        self,
        insertion_electrode: MongoStore,
        migration_graph: MongoStore,
        rank_electrode: MongoStore,
        **kwargs,
    ):
        self.insertion_electrode = insertion_electrode
        self.migration_graph = migration_graph
        self.rank_electrode = rank_electrode
        super().__init__(source=insertion_electrode, target=rank_electrode, **kwargs)
        self.sources.append(migration_graph)
        self.connect()

        self.mpr = MPRester()  # doesn't work for emmet

    def get_voltage_cost(self, v, lower_x_cutoff=1.8, upper_x_cutoff=2.8, cost_trans=1):
        x_target = (lower_x_cutoff + upper_x_cutoff) / 2
        cost = cost_trans * ((v - x_target) / (lower_x_cutoff - x_target)) ** 2
        return cost

    def get_stability_cost(self, v, x_trans=0.2, cost_trans=2):
        cost = cost_trans * (2 * np.e ** (4 * (v - x_trans)) - 1)
        if cost < 0:
            cost = 0
        return cost

    def get_items(self):
        for item in super(RankElecBuilder, self).get_items():
            mg_doc = self.migration_graph.query_one(
                {self.migration_graph.key: item[self.insertion_electrode.key]}
            )
            item.update({"mg_doc": mg_doc})
            yield item

    def unary_function(self, item: dict) -> dict:

        # check if ICSD experimental structure
        struct = Structure.from_dict(item["host_structure"])
        mp_ids = self.mpr.find_structure(struct)
        icsd_exp = False
        icsd_ids = []
        for q in self.mpr.query(
            {"material_id": {"$in": mp_ids}}, ["theoretical", "icsd_ids"]
        ):
            if q["theoretical"] == False and len(q["icsd_ids"]) > 0:
                icsd_exp = True
                icsd_ids.extend(q["icsd_ids"])

        # calculate costs
        v_cost = self.get_voltage_cost(item["average_voltage"])
        chg_stability_cost = self.get_stability_cost(item["stability_charge"])
        dchg_stability_cost = self.get_stability_cost(item["stability_discharge"])
        cost = v_cost + chg_stability_cost + dchg_stability_cost

        # removed unneeded fields from insertion_electrode store doc
        out = item.copy()
        for k in ["warnings", "last_updated", "adj_pairs", "_id"]:
            if k in out.keys():
                del out[k]

        # store additional fields
        out["host_mp_ids"] = mp_ids
        out["icsd_experimental"] = icsd_exp
        out["host_icsd_ids"] = icsd_ids
        out["cost"] = {
            "total": cost,
            "voltage": v_cost,
            "chg_stability": chg_stability_cost,
            "dchg_stability": dchg_stability_cost,
        }

        # if available add migration graph information
        mg_doc_keys = [
            "hop_cutoff",
            "entries_for_generation",
            "working_ion_entry",
            "migration_graph",
            "populate_sc_fields",
            "matrix_supercell_structure",
            "conversion_matrix",
            "inserted_ion_coords",
            "insert_coords_combo",
        ]
        if item["mg_doc"] is None:
            migration_graph_found = False
        elif item["mg_doc"]["state"] == "failed":
            migration_graph_found = False
        else:
            migration_graph_found = True
            for k in mg_doc_keys:
                out.update({k: item["mg_doc"][k]})

            # check for pathway connectivity
            if item["mg_doc"]["migration_graph"] is None:
                num_paths_found = None
            else:
                mgd = MigrationGraphDoc.parse_obj(item["mg_doc"])
                mgd.migration_graph.assign_cost_to_graph()
                num_paths_found = 0
                for n, hops in mgd.migration_graph.get_path():
                    num_paths_found += 1

        if migration_graph_found is False:
            num_paths_found = None
            for k in mg_doc_keys:
                out.update({k: None})

        out["migration_graph_found"] = migration_graph_found
        out["num_paths_found"] = num_paths_found
        del out["mg_doc"]

        return out
