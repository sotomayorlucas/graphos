"""RouterSink — multi-input terminal node that dispatches packets to actions."""

import threading

from dataflow.node import Node
from dataflow.primitives import _STOP


class RouterSink(Node):
    """Reads class IDs from 'classes' and raw packets from 'packets',
    dispatches each packet to action handlers based on its class ID."""

    def __init__(self, name: str, actions: dict, default_action=None):
        super().__init__(name)
        self.add_input("classes")
        self.add_input("packets")
        self._actions = actions          # dict[int, list[Action]]
        self._default_action = default_action
        self.results: list = []
        self._lock = threading.Lock()

    def process(self):
        classes_in = self.inputs["classes"]
        packets_in = self.inputs["packets"]

        while True:
            batch = classes_in.get()

            if batch is _STOP:
                # Drain packets channel until _STOP
                while True:
                    pkt = packets_in.get()
                    if pkt is _STOP:
                        break
                self._close_actions()
                return

            # batch is a list of class IDs (one per packet in the batch)
            for class_id in batch:
                pkt = packets_in.get()
                if pkt is _STOP:
                    # Unexpected early stop on packets channel
                    self._close_actions()
                    return

                handlers = self._actions.get(class_id, [])
                for action in handlers:
                    action.execute(pkt, class_id)
                if not handlers and self._default_action is not None:
                    self._default_action.execute(pkt, class_id)

                with self._lock:
                    self.results.append(class_id)

            self.items_processed += len(batch)

    def _close_actions(self):
        """Call close() on all registered actions."""
        seen = set()
        for handlers in self._actions.values():
            for action in handlers:
                if id(action) not in seen:
                    seen.add(id(action))
                    action.close()
        if self._default_action is not None and id(self._default_action) not in seen:
            self._default_action.close()

    def teardown(self):
        self._close_actions()
