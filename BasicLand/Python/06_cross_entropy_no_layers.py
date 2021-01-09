import math
# Loss inside a net from cross-entropy calculation

netOutput = [0.88, 0.24, 0.14]
targetOutput = [1, 0, 0]

loss = -(math.log(netOutput[0])*targetOutput[0] +
         math.log(netOutput[1])*targetOutput[1] +
         math.log(netOutput[2])*targetOutput[2] )

print("  Loss: ", loss)
