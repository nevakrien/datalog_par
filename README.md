# datalog_par
a datalog implementation thats standard complient and fast.

the goal is to try and see if its possible to make an effishent engine for this languge. the target computer is 32+ cores and this matters becuase some algorithems actually benifit from spliting diffrently when you have 2-4 cores.
we also want to work well on any workload, for now I am naively parallalizing anything that can be.
there is a strong possibility that most programs would benifit from less parallalisem in key points.

in terms of the parser it seems to be complient.

# standard
so datalog does not have a proper standard other than being a subset of prolog.
I am aiming for ISO compliance but I dont own a document. however here
https://www.swi-prolog.org/pldoc/man?section=isosyntax

we have a pretty good idea on how atoms and such should look.
and so this + what i could get from chatbots and articles is used as a refrence.

any non compliance is considered a bug that should be reported.

# TODO
1. add example programs
2. figure out a better way to name files
3. write a proper compile step

