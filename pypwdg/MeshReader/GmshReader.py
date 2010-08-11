import numpy

def GmshReader(fname):

    meshDict={}

    def ReadVersion(content):
        it=iter(content)
        while ("$MeshFormat"!=it.next()): pass
        meshDict['MeshFormat']=it.next().split()[0]

    def ReadNodes(content):
        nodes=[]
        it=iter(content)
        while ("$Nodes"!=it.next()): pass
        nnodes=int(it.next())
        for i in range(nnodes):
            line=it.next().split()
            nodes.append(numpy.array(map(float,line[1:])))
        meshDict['nodes']=nodes
        meshDict['nnodes']=nnodes

    def ReadElements(content):
        elements={}
        it=iter(content)
        while ("$Elements"!=it.next()): pass
        nelements=int(it.next())
        for i in range(nelements):
            line=it.next().split()
            id=int(line[0])
            elemtype=int(line[1])
            ntags=int(line[2])
            tags=map(int,line[3:3+ntags])
            physEntity=tags[0]
            geomEntity=tags[1]
            meshPartition=tags[2]
            nodes=map(int,line[3+ntags:])
            nodes=[i-1 for i in nodes]
            elements[id]={'type':elemtype,
                          'id':id,
                          'physEntity':physEntity,
                          'geomEntity':geomEntity,
                          'meshPartition':meshPartition,
                          'tags':tags,
                          'nodes':nodes}
        meshDict['elements']=elements
        meshDict['nelements']=nelements


    input=open(fname)
    content=input.read().split('\n')    
    ReadVersion(content)
    ReadNodes(content)
    ReadElements(content)
    return meshDict

if __name__ == "__main__":
    mesh=TriangleMesh2D('/Users/tbetcke/Desktop/square.msh')





