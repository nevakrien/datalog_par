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

# profiling
runing something like flamegraph gives completly useless results because of rayon.
vtune is better suited for this sort of work and we get that most things were inlined and most work is 
1. vec allocs
2. comperison
3. hashing
4. cross beam stuff

which seems close to the worse situation we could possibly get.
this is on a benchmark which we should be best suited for.

so there is a real need to reduce the amount of rayon we do in favor of just runin directly.


we also most likely want to add instromentation.
I have a few things in mind but a simple inline(never) on a few things would already go a long way


```bash
vtune -report exec-query -r vtune_results \
  -rep-knob row-by="/CPUFunction" \
  -rep-knob column-by="CPUFunctionModule|CPUTimeSummary|CPUTimeSummaryPercentage" \
  -sort-desc "CPU Time:Self" -format csv \
| awk -F, '
  BEGIN {
    print "ShortName,Module,CPUTime_Self,Percent_Self,FullFunction,CumPercent"
  }
  NR==1 {
    for (i=1;i<=NF;i++) {
      if($i ~ /^Function$/) f=i
      if($i ~ /^Module$/) m=i
      if($i ~ /^CPU Time:Self$/) ts=i
      if($i ~ /^% of CPU Time:Self/) ps=i
    }
    next
  }
  NR>1 && $f != "" {
    gsub(/%/,"",$ps)
    perc = ($ps+0)
    cum += perc
    short=$f
    sub(/.*::/,"",short)
    # one single CSV line with exactly 6 fields
    printf "%s,%s,%s,%s,%.2f%%,%.2f%%\n", short,$f,$m,$ts,perc,cum
  }' > vtune_summary.csv

```

```bash
vtune -report exec-query -r vtune_results \
  -rep-knob row-by="/CPUFunction" \
  -rep-knob column-by="CPUFunctionModule|CPUTimeSummary|CPUTimeSummaryPercentage" \
  -sort-desc "CPU Time:Self" -format csv \
| awk -F, '
  BEGIN {
    print "ShortName,Module,CPUTime_Self,Percent_Self"
  }
  NR==1 {
    for (i=1;i<=NF;i++) {
      if($i ~ /^Function$/) f=i
      if($i ~ /^Module$/) m=i
      if($i ~ /^CPU Time:Self$/) ts=i
      if($i ~ /^% of CPU Time:Self/) ps=i
    }
    next
  }
  NR>1 && $f != "" {
    gsub(/%/,"",$ps)
    perc = ($ps+0)
    short=$f
    sub(/.*::/,"",short)
    # one single CSV line with exactly 6 fields
    printf "%s,%s,%s,%.2f%%\n", short,$f,$m,perc
  }' > vtune_summary.csv

```