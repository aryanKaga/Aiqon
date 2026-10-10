from pathlib import Path
import pandas as pd

# <project root>/Data/data  (this file lives in <project root>/Recommendation_Engine/)
DATA_DIR = Path(__file__).resolve().parent.parent / "Data" / "data"


class DataLoader:
    def __init__(self, data_dir=DATA_DIR):
        data_dir = Path(data_dir)
        self.legs = pd.read_csv(data_dir / "legs.csv").set_index("leg_id", drop=False)
        self.routes = pd.read_csv(data_dir / "routes.csv")
        self.shipments = pd.read_csv(data_dir / "shipments.csv").set_index("shipment_id", drop=False)
        self.hubs = pd.read_csv(data_dir / "hubs.csv")
        self.graph = self._build_graph()

    def _build_graph(self):
        """Adjacency list: from_hub -> list of edges. Each edge is a dict with every column of legs.csv
        (leg_id, from_hub, to_hub, mode, distance_km, costs, transit_days, delay_*, congestion_*, geo_risk_*),
        so parallel sea / rail / air edges between the same two hubs stay separate."""
        g = {}
        for edge in self.legs.to_dict("records"):
            g.setdefault(edge["from_hub"], []).append(edge)
        return g

    def shipment(self, shipment_id: str) -> pd.Series:
        if shipment_id not in self.shipments.index:
            raise KeyError(f"No shipment with id {shipment_id}")
        return self.shipments.loc[shipment_id]

    def route(self, route_id: str) -> pd.Series:
        match = self.routes[self.routes.route_id == route_id]
        if match.empty:
            raise KeyError(f"No route with id {route_id}")
        return match.iloc[0]

    def route_legs(self, route_id: str) -> pd.DataFrame:
        return self.legs.loc[self.route(route_id).leg_ids.split("|")]

    def candidates(self, shipment_id_or_origin, destination=None) -> pd.DataFrame:
        """Routes of the shipment's origin-destination pair (routes.csv is independent of shipments)."""
        if destination is None:
            s = self.shipment(shipment_id_or_origin)
            origin, dest = s.origin, s.destination
        else:
            origin, dest = shipment_id_or_origin, destination
        return self.routes[(self.routes.origin == origin) &
                           (self.routes.destination == dest)]