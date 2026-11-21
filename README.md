# Afterglow

Afterglow chooses destinations from the next game circle, visible pickups and supply drops. It scores travel time against the circle timer, nearby avatars, recovery opportunities, and the value of the next upgrade. It reevaluates each observation and uses its progress history to recover from blocked moves.

Build with `docker build -t afterglow .`. From the official kit, run `python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" afterglow`.

The standard-library JSON-lines wrapper, `mechfight.py`, is supplied in the [Builders starter kit](https://www.joinbuilders.org/buildathon/tracks/builders). The decision policy is original. It uses the game's documented visible-object interaction and the Docker runtime from the starter. Official practice uses 48 private matches. The local conformance stream checks protocol correctness and response time.
