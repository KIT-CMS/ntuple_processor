import json
import time
from pathlib import Path

try:
    import logging

    from config.logging_setup_configs import setup_logging
    logger = setup_logging(logger=logging.getLogger(__name__))
except ModuleNotFoundError:
    logger = logging.getLogger(__name__)


class CutOrderingManager:
    def __init__(self, cache_path=None, discover=False):
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.discover = discover

        self._importance_by_dataset = {}
        self._cutflow_stats_by_dataset = {}

    def load_cache(self):
        if self.cache_path is None:
            return
        if not self.cache_path.exists():
            return

        try:
            with self.cache_path.open("r", encoding="utf-8") as _file:
                payload = json.load(_file)

            raw_datasets = payload.get("datasets", {}) if isinstance(payload, dict) else {}
            if not isinstance(raw_datasets, dict):
                logger.warning(f"Ignoring malformed cut-order cache at {self.cache_path}")
                return

            loaded = {}
            for dataset_name, dataset_order in raw_datasets.items():
                if not isinstance(dataset_order, dict):
                    continue
                loaded[str(dataset_name)] = {
                    str(cut_name): float(importance)
                    for cut_name, importance in dataset_order.items()
                    if isinstance(cut_name, str) and isinstance(importance, (int, float))
                }

            self._importance_by_dataset = loaded
            logger.info(
                f"Loaded cut-order cache for {len(self._importance_by_dataset)} dataset(s) from {self.cache_path}"
            )
        except Exception as err:
            logger.warning(f"Failed to load cut-order cache from {self.cache_path}: {err}")

    def order_cuts(self, cuts, dataset_name=None):
        if dataset_name is None:
            return list(cuts)

        importance = self._importance_by_dataset.get(str(dataset_name), {})
        if not importance:
            return list(cuts)

        # Unknown cuts are moved to the end while preserving relative order.
        return sorted(
            list(cuts),
            key=lambda cut: importance.get(cut.name, float("-inf")),
            reverse=True,
        )

    def collect_cutflow_report(self, dataset_name, reports):
        if not self.discover or dataset_name is None:
            return {}

        dataset_name = str(dataset_name)
        merged_stats = {dataset_name: {}}
        for report in reports:
            for cut_info in report:
                cut_name = str(cut_info.GetName())
                all_events = int(cut_info.GetAll())
                pass_events = int(cut_info.GetPass())
                if cut_name not in merged_stats[dataset_name]:
                    merged_stats[dataset_name][cut_name] = {"all": 0, "pass": 0}
                merged_stats[dataset_name][cut_name]["all"] += all_events
                merged_stats[dataset_name][cut_name]["pass"] += pass_events

        return merged_stats

    def merge_cutflow_stats(self, merged_stats):
        for dataset_name, cut_map in merged_stats.items():
            dataset_stats = self._cutflow_stats_by_dataset.setdefault(dataset_name, {})
            for cut_name, stats in cut_map.items():
                cut_stats = dataset_stats.setdefault(cut_name, {"all": 0, "pass": 0})
                cut_stats["all"] += int(stats.get("all", 0))
                cut_stats["pass"] += int(stats.get("pass", 0))

    def finalize_discovery(self):
        if not self.discover:
            return

        for dataset_name, cut_stats in self._cutflow_stats_by_dataset.items():
            dataset_order = self._importance_by_dataset.get(dataset_name, {})
            for cut_name, stats in cut_stats.items():
                all_events = stats["all"]
                pass_events = stats["pass"]
                if all_events <= 0:
                    continue
                fail_events = max(all_events - pass_events, 0)
                dataset_order[cut_name] = fail_events / float(all_events)
            if dataset_order:
                self._importance_by_dataset[dataset_name] = dataset_order

        self._save_cache()

    def _save_cache(self):
        if self.cache_path is None:
            logger.warning("discover_cut_ordering is enabled but no cut_order_cache_path is set")
            return

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "datasets": self._importance_by_dataset,
        }

        with self.cache_path.open("w", encoding="utf-8") as _file:
            json.dump(payload, _file, indent=2, sort_keys=True)
            _file.write("\n")

        logger.info(f"Wrote cut-order cache to {self.cache_path}")
