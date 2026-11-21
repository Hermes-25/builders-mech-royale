"""Walkable tiles for reaching game-zone and reward markers."""
import heapq
import math


def intervals(ring, y):
    hits=[]
    x0,y0=ring[-2:]
    for x1,y1 in zip(ring[::2],ring[1::2]):
        if (y0 <= y < y1) or (y1 <= y < y0):
            hits.append(x0+(y-y0)*(x1-x0)/(y1-y0))
        x0,y0=x1,y1
    hits.sort()
    return list(zip(hits[::2],hits[1::2]))


class TileRoutes:
    def __init__(self, bounds, prisms, ground, mech, step=6):
        self.x=bounds["minX"]
        self.y=bounds["minY"]
        self.step=step
        self.w=math.ceil((bounds["maxX"]-self.x)/step)
        self.h=math.ceil((bounds["maxY"]-self.y)/step)
        self.open=bytearray([1])*(self.w*self.h)
        # Sample columns across each footprint row. Ignore overhead platforms
        # with enough clearance and steps the avatar can walk onto.
        for solid in prisms:
            ring=solid["ring"]
            ys=ring[1::2]
            if not ys:continue
            j0=max(0,math.floor((min(ys)-self.y)/step))
            j1=min(self.h-1,math.floor((max(ys)-self.y)/step))
            for j in range(j0,j1+1):
                y=self.y+(j+0.5)*step
                holes=[span for hole in solid.get("holes",()) for span in intervals(hole,y)]
                for left,right in intervals(ring,y):
                    i0=max(0,math.ceil((left-self.x)/step-0.5))
                    i1=min(self.w-1,math.floor((right-self.x)/step-0.5))
                    for i in range(i0,i1+1):
                        x=self.x+(i+0.5)*step
                        if any(a<x<b for a,b in holes):continue
                        z=ground(x,y)
                        if solid["base"] < z+mech["height"] and solid["top"] > z+mech["step_up"]:
                            self.open[j*self.w+i]=0
        self.route=[]
        self.last_goal=None
        self.checked=-100

    def cell(self,p):
        i=min(self.w-1,max(0,int((p[0]-self.x)//self.step)))
        j=min(self.h-1,max(0,int((p[1]-self.y)//self.step)))
        return j*self.w+i

    def point(self,node):
        j,i=divmod(node,self.w)
        return self.x+(i+0.5)*self.step,self.y+(j+0.5)*self.step

    def nearest(self,node):
        if self.open[node]:return node
        j,i=divmod(node,self.w)
        for r in range(1,6):
            opts=[(di*di+dj*dj,(j+dj)*self.w+i+di)
                  for dj in range(-r,r+1) for di in range(-r,r+1)
                  if max(abs(di),abs(dj))==r and 0<=i+di<self.w and 0<=j+dj<self.h
                  and self.open[(j+dj)*self.w+i+di]]
            if opts:return min(opts)[1]
        return None

    def clear(self,a,b):
        ay,ax=divmod(a,self.w);by,bx=divmod(b,self.w)
        dx,dy=bx-ax,by-ay
        sx,sy=(1 if dx>0 else -1),(1 if dy>0 else -1)
        nx,ny=abs(dx),abs(dy)
        x,y=ax,ay;ix=iy=0
        while ix<nx or iy<ny:
            cross=(1+2*ix)*ny-(1+2*iy)*nx
            if cross==0:
                if not self.open[y*self.w+x+sx] or not self.open[(y+sy)*self.w+x]:return False
                x+=sx;y+=sy;ix+=1;iy+=1
            elif cross<0:x+=sx;ix+=1
            else:y+=sy;iy+=1
            if not self.open[y*self.w+x]:return False
        return True

    def search(self,start,goal):
        gy,gx=divmod(goal,self.w)
        def h(node):
            j,i=divmod(node,self.w);dx,dy=abs(i-gx),abs(j-gy)
            return 10*max(dx,dy)+4*min(dx,dy)
        cost={start:0};parent={};queue=[(h(start),start,0)]
        best=start
        for _ in range(1850):
            if not queue:break
            _,node,g=heapq.heappop(queue)
            if g!=cost[node]:continue
            if h(node)<h(best):best=node
            if node==goal:break
            j,i=divmod(node,self.w)
            for di,dj in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                ni,nj=i+di,j+dj
                if not (0<=ni<self.w and 0<=nj<self.h):continue
                adj=nj*self.w+ni
                if not self.open[adj]:continue
                if di and dj and (not self.open[j*self.w+ni] or not self.open[nj*self.w+i]):continue
                ng=g+(14 if di and dj else 10)
                if ng>=cost.get(adj,10**9):continue
                cost[adj]=ng;parent[adj]=node
                heapq.heappush(queue,(ng+h(adj),adj,ng))
        nodes=[]
        while best!=start:
            nodes.append(best);best=parent[best]
        return nodes[::-1]

    def open_segment(self,p,q):
        a,b=self.cell(p),self.cell(q)
        if not self.open[a] or not self.open[b]:return False
        if a==b:return True
        iy,ix=divmod(a,self.w);jy,jx=divmod(b,self.w)
        x=(p[0]-self.x)/self.step; y=(p[1]-self.y)/self.step
        dx=(q[0]-p[0])/self.step;dy=(q[1]-p[1])/self.step
        sx=1 if dx>0 else -1;sy=1 if dy>0 else -1
        tx=((ix+(sx>0))-x)/dx if dx else math.inf
        ty=((iy+(sy>0))-y)/dy if dy else math.inf
        tx_step=abs(1/dx) if dx else math.inf
        ty_step=abs(1/dy) if dy else math.inf
        for _ in range(abs(jx-ix)+abs(jy-iy)+1):
            if ix==jx and iy==jy:return True
            if abs(tx-ty)<1e-8:
                if not self.open[iy*self.w+ix+sx] or not self.open[(iy+sy)*self.w+ix]:return False
                ix+=sx;iy+=sy;tx+=tx_step;ty+=ty_step
            elif tx<ty:ix+=sx;tx+=tx_step
            else:iy+=sy;ty+=ty_step
            if not (0<=ix<self.w and 0<=iy<self.h and self.open[iy*self.w+ix]):return False
        return False

    def waypoint(self,p,goal,now):
        a=self.nearest(self.cell(p));b=self.nearest(self.cell(goal))
        if a is None or b is None:return goal
        if self.open_segment(p,goal):
            self.route.clear()
            return goal
        too_far=self.last_goal is None or math.dist(goal[:2],self.last_goal)>2*self.step
        if (now-self.checked>=1.4 and (too_far or not self.route)
                or now-self.checked>6.0):
            self.route=self.search(a,b)
            self.last_goal=tuple(goal[:2]);self.checked=now
        # Skip already visited and mutually visible intermediate cells.
        if not self.route:return goal
        furthest=None
        for idx,node in enumerate(self.route[:22]):
            if self.open_segment(p,self.point(node)):furthest=idx
            else:break
        if furthest is None:
            self.route.clear()
            return self.point(a)
        if furthest:
            self.route=self.route[furthest:]
        if len(self.route)==1 and self.route[0]==a:
            self.route.clear();return goal
        return self.point(self.route[0])
