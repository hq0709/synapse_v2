#!/usr/bin/env python3
"""
Synapse 2.0 - Memory-Driven Scientific Exploration
Two core functions: Memory (store/ask/trace) and Explore (deep reasoning)
"""

import sys
import os
import time
import threading
import itertools
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.synapse_brain import SynapseBrain


class C:
    """Colors - blue/white only"""
    B = '\033[38;5;39m'    # Blue
    L = '\033[38;5;117m'   # Light blue
    D = '\033[38;5;153m'   # Dim blue
    W = '\033[97m'         # White
    BOLD = '\033[1m'
    R = '\033[0m'          # Reset


class Spinner:
    """Animated spinner for LLM thinking phases"""
    FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    def __init__(self, text="Thinking"):
        self.text = text
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def _spin(self):
        for frame in itertools.cycle(self.FRAMES):
            if not self._running:
                break
            with self._lock:
                text = self.text
            sys.stdout.write(f"\r{C.D}  {frame} {text}{C.R}  ")
            sys.stdout.flush()
            time.sleep(0.08)

    def update(self, text):
        with self._lock:
            self.text = text

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        sys.stdout.write(f"\r{' ' * 70}\r")
        sys.stdout.flush()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


class Synapse:
    def __init__(self):
        clear()
        print(f"\n{C.B}{C.BOLD}  SYNAPSE 2.0{C.R}")
        print(f"{C.L}  Memory-Driven Scientific Exploration{C.R}\n")

        self.brain = SynapseBrain()

    def run(self):
        self._show_help()

        while True:
            try:
                user_input = input(f"{C.B}>{C.R} ").strip()

                if not user_input:
                    continue

                if user_input.startswith('/'):
                    self._handle_command(user_input[1:])
                else:
                    # Direct input = ask question
                    self._ask(user_input)

            except KeyboardInterrupt:
                print(f"\n{C.D}Use /exit to quit{C.R}")
            except EOFError:
                break

    def _show_help(self):
        print(f"{C.B}Commands:{C.R}")
        print(f"  {C.L}/memory upload <file>{C.R}  Upload document")
        print(f"  {C.L}/memory trace <query>{C.R}  Trace memory")
        print(f"  {C.L}/explore <topic>{C.R}       Deep exploration")
        print(f"  {C.L}/status{C.R}                Show stats")
        print(f"  {C.L}/selfcheck{C.R}             Preflight diagnostics")
        print(f"  {C.L}/help{C.R}                  Commands")
        print(f"  {C.L}/exit{C.R}                  Quit")
        print(f"\n{C.D}Or type a question directly{C.R}\n")

    def _handle_command(self, cmd_str: str):
        parts = cmd_str.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd in ('m', 'memory'):
            self._memory_cmd(args)
        elif cmd in ('e', 'explore'):
            self._explore(args)
        elif cmd in ('s', 'status'):
            self._status()
        elif cmd in ('selfcheck', 'check'):
            self._selfcheck()
        elif cmd in ('h', 'help', '?'):
            self._show_help()
        elif cmd in ('c', 'clear'):
            clear()
        elif cmd in ('exit', 'quit', 'q'):
            self._exit()
        else:
            print(f"{C.D}Unknown: /{cmd}{C.R}")

    # ---- Memory commands ----

    def _memory_cmd(self, args: str):
        if not args:
            print(f"\n{C.B}Memory:{C.R}")
            print(f"  /memory upload <file>  Upload document")
            print(f"  /memory trace <query>  Trace connections")
            print(f"  /memory stats          Statistics\n")
            return

        parts = args.split(maxsplit=1)
        sub = parts[0].lower()
        sub_args = parts[1] if len(parts) > 1 else ""

        if sub == 'upload':
            self._upload(sub_args)
        elif sub == 'trace':
            self._trace(sub_args)
        elif sub == 'stats':
            self._status()
        else:
            print(f"{C.D}Unknown: /memory {sub}{C.R}")

    def _upload(self, filepath: str):
        if not filepath:
            print(f"{C.D}Usage: /memory upload <filepath>{C.R}")
            return

        if not os.path.exists(filepath):
            print(f"{C.D}File not found: {filepath}{C.R}")
            return

        spinner = Spinner("Reading document")
        spinner.start()

        def on_status(phase):
            spinner.update(phase)

        result = self.brain.upload(filepath, status_callback=on_status)
        spinner.stop()

        if 'error' in result:
            print(f"{C.D}Error: {result['error']}{C.R}")
            return

        print(f"{C.B}Done:{C.R}")
        print(f"  Facts extracted: {result['memcells_created']}")
        print(f"  Episodes created: {result['episodes_created']}")
        print(f"  Topics identified: {', '.join(result.get('topics', []))}\n")

    def _ask(self, question: str):
        spinner = Spinner("Retrieving memories")
        spinner.start()

        def on_status(phase):
            spinner.update(phase)

        result = self.brain.ask(question, status_callback=on_status)
        spinner.stop()

        if 'error' in result:
            print(f"\n{C.D}Error: {result['error']}{C.R}\n")
            return

        print(f"\n{C.B}Answer:{C.R}")
        print(result['answer'])

        # Metadata line
        meta_parts = []
        if result['memories_used'] > 0:
            meta_parts.append(f"Memories: {result['memories_used']}")
        meta_parts.append(f"{result['elapsed']:.1f}s")
        if result.get('mode'):
            meta_parts.append(result['mode'])
        print(f"\n{C.D}{' | '.join(meta_parts)}{C.R}")

        if result.get('sources'):
            print(f"{C.D}Sources: {', '.join(result['sources'][:3])}{C.R}")

        contract = result.get('evidence_contract') or {}
        coverage = contract.get('citation_coverage') or {}
        if coverage:
            cited = coverage.get('evidence_items_cited', 0)
            total = coverage.get('evidence_items_total', 0)
            sent_cited = coverage.get('sentences_with_citation', 0)
            sent_total = coverage.get('sentences_total', 0)
            print(f"{C.D}Evidence contract: cited evidence {cited}/{total} | citation sentences {sent_cited}/{sent_total}{C.R}")
        if contract.get('reasoning_risks'):
            print(f"{C.D}Risks: {contract['reasoning_risks'][0]}{C.R}")

        print()

    def _trace(self, query: str):
        if not query:
            print(f"{C.D}Usage: /memory trace <query>{C.R}")
            return

        spinner = Spinner("Tracing memories")
        spinner.start()
        trace = self.brain.trace(query)
        spinner.stop()

        if not trace.get('found'):
            print(f"{C.D}No memories found{C.R}\n")
            return

        for i, t in enumerate(trace['traces'][:5]):
            print(f"\n{C.B}[{i}] {t['type'].upper()}{C.R} (relevance: {t['relevance']:.4f})")
            print(f"  {C.W}{t['content'][:150]}{C.R}")
            print(f"  {C.D}Source: {t['source']}{C.R}")

            if t['evidence']:
                for ev in t['evidence'][:2]:
                    print(f"  {C.D}Evidence: {ev['content'][:80]}... [{ev['type']}, conf: {ev['confidence']:.0%}]{C.R}")

            if t['connections'] > 0:
                print(f"  {C.D}Connections: {t['connections']}{C.R}")

        if trace.get('related_episodes'):
            print(f"\n{C.B}Related episodes:{C.R}")
            for ep in trace['related_episodes'][:3]:
                print(f"  {C.L}• {ep['subject']}{C.R}")
                print(f"    {C.D}{ep['summary'][:100]}{C.R}")

        print()

    # ---- Explore ----

    def _explore(self, args: str):
        if not args:
            print(f"{C.D}Usage: /explore <topic> [depth:N]{C.R}")
            return

        # Parse depth
        parts = args.rsplit('depth:', 1)
        topic = parts[0].strip()
        depth = 3
        if len(parts) > 1:
            try:
                depth = max(1, min(5, int(parts[1].strip())))
            except ValueError:
                pass

        print(f"\n{C.B}{C.BOLD}Deep Exploration: {topic}{C.R}")
        print(f"{C.D}Depth: {depth} iterations{C.R}\n")

        spinner = Spinner("Reviewing current knowledge")
        spinner.start()

        def on_status(phase):
            spinner.update(phase)

        result = self.brain.explore(topic, depth=depth, status_callback=on_status)
        spinner.stop()

        if 'error' in result:
            print(f"\n{C.D}Error: {result['error']}{C.R}\n")
            return

        # Display results
        for it in result['iterations']:
            n = it['iteration']
            print(f"\n{C.B}--- Iteration {n} ({it.get('duration', 0):.1f}s) ---{C.R}")

            if it.get('gaps'):
                print(f"{C.L}Knowledge gaps:{C.R}")
                for gap in it['gaps'][:3]:
                    if isinstance(gap, str):
                        print(f"  {C.W}{gap}{C.R}")

            if it.get('hypotheses'):
                print(f"{C.L}Hypotheses:{C.R}")
                for hyp in it['hypotheses'][:2]:
                    if isinstance(hyp, dict):
                        print(f"  {C.W}{hyp.get('hypothesis', str(hyp))}{C.R}")
                        if hyp.get('rationale'):
                            print(f"    {C.D}{hyp['rationale'][:120]}{C.R}")
                    elif isinstance(hyp, str):
                        print(f"  {C.W}{hyp}{C.R}")

            if it.get('ranked_hypotheses'):
                print(f"{C.L}Prioritized hypotheses:{C.R}")
                for hyp in it['ranked_hypotheses'][:2]:
                    if isinstance(hyp, dict):
                        print(f"  {C.W}{hyp.get('hypothesis', '')[:130]}{C.R}")
                        p = hyp.get('priority_score', 0.0)
                        t = hyp.get('testability_score', 0.0)
                        f = hyp.get('falsifiability_score', 0.0)
                        print(f"    {C.D}priority={p:.2f}, testability={t:.2f}, falsifiability={f:.2f}{C.R}")

            if it.get('experiments'):
                print(f"{C.L}Experiment plans:{C.R}")
                for exp in it['experiments'][:2]:
                    if isinstance(exp, dict):
                        print(f"  {C.W}{exp.get('experiment', '')[:140]}{C.R}")
                        if exp.get('measurable_outcome'):
                            print(f"    {C.D}Metric: {exp['measurable_outcome'][:100]}{C.R}")
                        if exp.get('failure_signal'):
                            print(f"    {C.D}Failure signal: {exp['failure_signal'][:100]}{C.R}")

            if it.get('feedback') and isinstance(it['feedback'], dict):
                fb = it['feedback']
                if fb.get('strongest'):
                    print(f"{C.L}Strongest:{C.R} {fb['strongest'][:150]}")

            if it.get('insights'):
                print(f"{C.L}Insights:{C.R}")
                for ins in it['insights'][:2]:
                    if isinstance(ins, str):
                        print(f"  {C.W}{ins}{C.R}")

        # Final synthesis
        if result.get('final_synthesis'):
            print(f"\n{C.B}{C.BOLD}Synthesis:{C.R}")
            print(result['final_synthesis'])

        print(f"\n{C.D}Duration: {result.get('total_duration', 0):.1f}s | Saved to memory{C.R}\n")

    # ---- Status ----

    def _status(self):
        stats = self.brain.get_stats()
        mem = stats['memory']

        print(f"\n{C.B}Brain Status:{C.R}")
        print(f"  LLM: {'Connected' if stats['llm_available'] else 'Offline'}")
        print(f"  Embeddings: {'Connected' if stats.get('embeddings_available') else 'Offline'}")
        print(f"  Vector backend: {stats.get('vector_backend', 'unknown')}")
        print(f"  LLM calls: {stats['llm_calls']}")
        print(f"  Memories: {mem['total_memcells']}")
        print(f"  Episodes: {mem['total_episodes']}")
        print(f"  Connections: {mem['total_connections']}")
        print(f"  Contradictions: {mem.get('total_contradictions', 0)}")
        print(f"  Retrievals: {mem['total_retrievals']}")
        print(f"  FAISS vectors: {stats.get('faiss_vectors', 0)}")
        print(f"  In-memory vectors: {stats.get('in_memory_vectors', 0)}")
        print(f"  Conversations: {stats['conversation_length']}")

        if stats['profiles']:
            print(f"  Topics: {', '.join(stats['profiles'][:5])}")

        print()

    def _selfcheck(self):
        checks = self.brain.preflight_check()
        print(f"\n{C.B}Preflight Check:{C.R}")
        print(f"  OK: {'Yes' if checks.get('ok') else 'No'}")
        print(f"  LLM: {'Connected' if checks.get('llm_available') else 'Offline'} ({checks.get('llm_model')})")
        print(f"  Embeddings: {'Connected' if checks.get('embedding_available') else 'Offline'} ({checks.get('embedding_model')})")
        print(f"  Vector backend: {checks.get('vector_backend')}")
        print(f"  FAISS vectors: {checks.get('faiss_vectors', 0)}")
        print(f"  In-memory vectors: {checks.get('in_memory_vectors', 0)}")
        print(f"  Run log dir writable: {'Yes' if checks.get('run_log_writable') else 'No'}")
        if checks.get('errors'):
            print(f"{C.D}Errors:{C.R}")
            for err in checks['errors']:
                print(f"  {C.D}- {err}{C.R}")
        print()

    def _exit(self):
        stats = self.brain.get_stats()
        print(f"\n{C.D}Memories: {stats['memory']['total_memcells']} | LLM calls: {stats['llm_calls']}{C.R}")
        self.brain.close()
        print(f"{C.L}Goodbye{C.R}\n")
        sys.exit(0)


def main():
    cli = Synapse()
    try:
        cli.run()
    except KeyboardInterrupt:
        cli._exit()


if __name__ == "__main__":
    main()
