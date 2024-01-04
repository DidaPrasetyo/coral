from pycoral.utils.edgetpu import list_edge_tpus, get_device_details

# List available Edge TPUs
edge_tpus = list_edge_tpus()
print("Available Edge TPUs:", edge_tpus)

# Get details about a specific Edge TPU
if edge_tpus:
    details = get_device_details(edge_tpus[0])
    print("Details for Edge TPU:", details)


