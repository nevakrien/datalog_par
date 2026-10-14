# datalog_par
a datalog implementation that's standard compliant and fast.

the goal is to try and see if it's possible to make an efficient engine for this language. the target computer is 32+ cores and this matters because some algorithms actually benefit from splitting differently when you have 2-4 cores.
we also want to work well on any workload, for now I am naively parallelizing anything that can be.
there is a strong possibility that most programs would benefit from less parallelism in key points.

in terms of the parser it seems to be compliant.

# standard
so datalog does not have a proper standard other than being a subset of prolog.
I am aiming for ISO compliance but I don't own a document. however here
https://www.swi-prolog.org/pldoc/man?section=isosyntax

we have a pretty good idea on how atoms and such should look.
and so this + what i could get from chatbots and articles is used as a reference.

any non compliance is considered a bug that should be reported.

# profiling
it has been a bit of a nightmare to get a proper profile. rayon has some absolutely terrible debug behavior.
perf and thus cargo flamegraph just can't handle it. vtune can handle it only for bottom up.
samply can do a very good job and after hours of debugging someone was kind enough to recommend it.

the following parts are the most expensive:
1. droping query solvers... just calling free there is 13% of runtime
2. writing to stdout is 26% of the time for just syscalls there is also 2% on formating
3. almost 22% of the time is spent just inserting to hashmaps
4. 37% of the time is spent just gathering data from rayon