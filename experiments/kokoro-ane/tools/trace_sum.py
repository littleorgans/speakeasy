"""Sums ANE and GPU busy time per process from an Instruments trace recorded with the Core ML template.
usage: trace_sum.py <trace> <process name>; needs xcrun xctrace. ANE rows carry the model and op label
(no process), so they count only while the named process ran; GPU rows carry the process."""
import collections, re, subprocess, sys
import xml.etree.ElementTree as ET


def table(trace, schema):
    xpath = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
    out = subprocess.run(["xcrun", "xctrace", "export", "--input", trace, "--xpath", xpath],
                         check=True, capture_output=True, text=True).stdout
    ids = {}
    for row in ET.fromstring(out).iter("row"):
        cells = {}
        for cell in row:
            if "ref" in cell.attrib:
                cell = ids[cell.attrib["ref"]]
            else:
                for node in cell.iter():
                    if "id" in node.attrib:
                        ids[node.attrib["id"]] = node
            cells.setdefault(cell.tag, cell)
        yield cells


def main(trace, process):
    ane = collections.Counter()
    count = collections.Counter()
    for c in table(trace, "ane-hw-intervals"):
        label = c["formatted-label"].attrib.get("fmt", "")
        # Multifunction models (the hybrid generator's 53 conv functions) collapse to the model.
        model = re.sub(r"_main__.*|__Op.*", "", label.split(" ")[0])
        model = re.sub(r"^(hybrid_generator_r\d+)_.*", r"\1", model)
        kind = "load" if " Load" in label else "predict"
        ane[(model, kind)] += int(c["duration"].text) / 1e6
        count[(model, kind)] += 1
    gpu = collections.Counter()
    for c in table(trace, "metal-gpu-intervals"):
        label = c["formatted-label"].attrib.get("fmt", "")
        if process in label:
            gpu[c["gpu-channel-name"].attrib.get("fmt", "?")] += int(c["duration"].text) / 1e6
    print("unit\tstage\tkind\tevents\tbusy ms")
    for (model, kind), ms in sorted(ane.items()):
        print(f"ANE\t{model}\t{kind}\t{count[(model, kind)]}\t{ms:.1f}")
    print(f"ANE\tall\tpredict\t{sum(n for (_, k), n in count.items() if k == 'predict')}\t"
          f"{sum(ms for (_, k), ms in ane.items() if k == 'predict'):.1f}")
    for channel, ms in sorted(gpu.items()):
        print(f"GPU\t{process}\t{channel}\t-\t{ms:.1f}")


main(*sys.argv[1:])
