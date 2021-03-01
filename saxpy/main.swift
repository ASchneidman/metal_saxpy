//
//  main.swift
//  saxpy
//
//  Created by Alex Schneidman on 2/28/21.
//

import Foundation
import Metal

let count = 1_000_000

let device = MTLCreateSystemDefaultDevice()!
let library = device.makeDefaultLibrary()!
let saxpy = library.makeFunction(name: "saxpy")!
let pipeline = try! device.makeComputePipelineState(function: saxpy)

print(device.name)

// Our data, randomly generated:

var x = [Float(0.0)]
var y = [Float(0.0)]

for _ in 0...count {
    x.append(Float.random(in: 1..<10))
}
for _ in 0...count {
    y.append(Float.random(in: 1..<10))
}

var alpha = Float(0.5)
var N = x.count

// Our data in a buffer (copied):
let x_buffer = device.makeBuffer(bytes: &x, length: MemoryLayout<Float>.stride * x.count, options: [])
let y_buffer = device.makeBuffer(bytes: &y, length: MemoryLayout<Float>.stride * y.count, options: [])
let alpha_buffer = device.makeBuffer(bytes: &alpha, length: MemoryLayout<Float>.stride, options: [])
let N_buffer = device.makeBuffer(bytes: &N, length: MemoryLayout<Int>.stride, options: [])

// A buffer for individual results (zero initialized)
let resultsBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * x.count, options: [])!
// Our results in convenient form to compute the actual result later:
let pointer = resultsBuffer.contents().bindMemory(to: Float.self, capacity: x.count)
let results = UnsafeBufferPointer<Float>(start: pointer, count: x.count)


let queue = device.makeCommandQueue()!
let cmds = queue.makeCommandBuffer()!
let encoder = cmds.makeComputeCommandEncoder()!

encoder.setComputePipelineState(pipeline)


encoder.setBuffer(x_buffer, offset: 0, index: 0)
encoder.setBuffer(y_buffer, offset: 0, index: 1)
encoder.setBuffer(resultsBuffer, offset: 0, index: 2)
encoder.setBuffer(N_buffer, offset: 0, index: 3)
encoder.setBuffer(alpha_buffer, offset: 0, index: 4)

// We have to calculate the sum `resultCount` times => amount of threadgroups is `resultsCount` / `threadExecutionWidth` (rounded up) because each threadgroup will process `threadExecutionWidth` threads
let threadgroupsPerGrid = MTLSize(width: (x.count + pipeline.threadExecutionWidth - 1) / pipeline.threadExecutionWidth, height: 1, depth: 1)

// Here we set that each threadgroup should process `threadExecutionWidth` threads, the only important thing for performance is that this number is a multiple of `threadExecutionWidth` (here 1 times)
let threadsPerThreadgroup = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)

encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
encoder.endEncoding()

var start, end : UInt64

start = mach_absolute_time()

cmds.commit()
cmds.waitUntilCompleted()

end = mach_absolute_time()

let gpu_time = Double(end - start) / Double(NSEC_PER_SEC)
print("Metal time: \(gpu_time)")

var cpu_results: [Float] = []
for _ in 0..<x.count {
    cpu_results.append(Float(0.0))
}
start = mach_absolute_time()
for index in 0..<x.count {
    cpu_results[index] = (alpha * x[index] + y[index])
}
end = mach_absolute_time()

let cpu_time = Double(end - start) / Double(NSEC_PER_SEC)
print("Cpu time: \(cpu_time)")

for index in 0..<x.count {
    if (abs(results[index] - cpu_results[index]) > Float(0.000005)) {
        print("index: \(index), metal result: \(results[index]), cpu result: \(alpha * x[index] + y[index])");
        exit(1)
    }
}
print("Verified results, relative speedup: \(cpu_time / gpu_time)")
exit(0)
