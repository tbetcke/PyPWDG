import numpy

def gmsh_reader(fname):

    meshDict={}

    def read_version(content):
        it=iter(content)
        while ("$MeshFormat"!=it.next()): pass
        meshDict['MeshFormat']=it.next().split()[0]

    def read_nodes(content):
        nodes=[]
        it=iter(content)
        while ("$Nodes"!=it.next()): pass
        nnodes=int(it.next())
        for i in range(nnodes):
            line=it.next().split()
            nodes.append(numpy.array(map(float,line[1:])))
        meshDict['nodes']=nodes
        meshDict['nnodes']=nnodes

    def read_elements(content):
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
    read_version(content)
    read_nodes(content)
    read_elements(content)
    return meshDict

if __name__ == "__main__":
    meshDict=gmsh_reader('../../examples/3D/cube.msh')





