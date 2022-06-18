using MiscUtils
using BenchmarkTools

c = 13 << 40;
@btime MiscUtils.findnzbits(13 << 40)		# < 1 ns
@btime MiscUtils.findnzbits(Val($c))		# 270 ns
@btime MiscUtils.findnzbits($c)				# 270 ns

@btime MiscUtils.findnzbits_(13 << 40)		# < 1 ns
@btime MiscUtils.findnzbits_(Val($c))		# 230 ns
@btime MiscUtils.findnzbits_($c)				# 230 ns


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
