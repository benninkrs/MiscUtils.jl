using MiscUtils
using BenchmarkTools

# Timings are for Julia v1.11.6 on HP ZBook (i7-1360P, 2.2GHz, 12 cores)
c = 13 << 40;
@btime MiscUtils.findnzbits(13 << 40)		# < 1 ns
@btime MiscUtils.findnzbits(Val($c))		# 66 ns
@btime MiscUtils.findnzbits($c)				# 66 ns

@btime MiscUtils.findnzbits_(13 << 40)		# < 1 ns
@btime MiscUtils.findnzbits_(Val($c))		# 66 ns
@btime MiscUtils.findnzbits_($c)				# 66 ns


function test(::Val{x}) where {x}
	local y
	for k = 1:1_000_000
		y = MiscUtils.findnzbits(x)
	end
	y
end


function test(x)
	local y
	for k = 1:1_000_000
		y = MiscUtils.findnzbits(x)
	end
	y
end


function test_(::Val{x}) where {x}
	local y
	for k = 1:1_000_000
		y = MiscUtils.findnzbits_(x)
	end
	y
end


function test_(x)
	local y
	for k = 1:1_000_000
		y = MiscUtils.findnzbits_(x)
	end
	y
end
