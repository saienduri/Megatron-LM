def compute_time(mbs, accum, time_comm, target=800.0, seq=4096.0):
    return (seq*mbs*accum/(target*8)-time_comm)/accum


print(compute_time(1, 4, 0.2))
print(compute_time(2, 4, 0.2))
print(compute_time(3, 4, 0.2))

