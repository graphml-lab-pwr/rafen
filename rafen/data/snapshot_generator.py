import datetime as dt

import networkx as nx
from dateutil.relativedelta import relativedelta


class GraphSnapshotDateGenerator:
    def __init__(self, graph, timestamp_format_year=False):
        self.graph = graph
        self.timestamp_format_year = timestamp_format_year
        self.snapshot_begin_date = None
        self.snapshot_end_date = None

        self.edgelist = sorted(
            graph.edges.data(), key=lambda edge: edge[2]["timestamp"]
        )
        self.begin_date = self.get_timestamp_from_edge(
            self.edgelist[0], self.timestamp_format_year
        )
        self.end_date = self.get_timestamp_from_edge(
            self.edgelist[-1], self.timestamp_format_year
        )

    @staticmethod
    def get_timestamp_from_edge(edge, timestamp_format_year=False):
        timestamp = edge[2]["timestamp"]
        if timestamp_format_year:
            return dt.datetime(year=int(timestamp), month=1, day=1)

        return dt.datetime.fromtimestamp(timestamp)

    def check_date_range(self, edge):
        timestamp = self.get_timestamp_from_edge(
            edge, self.timestamp_format_year
        )
        if self.snapshot_begin_date <= timestamp < self.snapshot_end_date:
            return True
        return False

    def generate(self, split_type, interval):
        if split_type == "year":
            interval = relativedelta(years=interval)
            self.snapshot_begin_date = dt.datetime(self.begin_date.year, 1, 1)
        elif split_type == "month":
            interval = relativedelta(months=interval)
            self.snapshot_begin_date = dt.datetime(
                self.begin_date.year, self.begin_date.month, 1
            )
        elif split_type == "day":
            interval = dt.timedelta(days=interval)
            self.snapshot_begin_date = dt.datetime(
                self.begin_date.year, self.begin_date.month, self.begin_date.day
            )
        elif split_type == "hour":
            interval = dt.timedelta(hours=interval)
            self.snapshot_begin_date = dt.datetime(
                year=self.begin_date.year,
                month=self.begin_date.month,
                day=self.begin_date.day,
                hour=self.begin_date.hour,
            )
        snapshot_id = 0
        while (self.snapshot_begin_date + interval) < (
            self.end_date + interval
        ):
            self.snapshot_end_date = self.snapshot_begin_date + interval
            snapshot_edgelist = [
                it for it in self.edgelist if self.check_date_range(it)
            ]
            self.snapshot_begin_date = self.snapshot_end_date

            snapshot = nx.from_edgelist(
                snapshot_edgelist, create_using=type(self.graph)
            )

            yield {
                "snapshot_id": snapshot_id,
                "graph": snapshot,
            }
            snapshot_id += 1
