#include <bf/bf.h>

#include <mfem.hpp>

using namespace mfem;
using namespace std;

void snap_nodes(Mesh *mesh);

int main(int argc, char *argv[]) {
  Mpi::Init(argc, argv);
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();

  int order = 5;
  int refine = 6;
  int nev = 10;
  int seed = 0;

  ConstantCoefficient one(1.0);
  ConstantCoefficient mu(1.0);

  // Set up our initial isoparametric cube mesh which we'll refine to
  // approximate the surface a sphere.

  Mesh *mesh = new Mesh(2, 8, 6, 0, 3);

  double verts[8][3] = {
    {-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1},
    {-1, -1, +1}, {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}
  };

  int faces[6][4] = {
    {3, 2, 1, 0}, {0, 1, 5, 4}, {1, 2, 6, 5},
    {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}
  };

  for (int i = 0; i < 8; ++i)
    mesh->AddVertex(verts[i]);

  for (int i = 0; i < 6; ++i) {
    int attrib = i + 1;
    mesh->AddQuad(faces[i], attrib);
  }

  mesh->FinalizeQuadMesh(1, 1, true);

  H1_FECollection fec(order, mesh->Dimension());
  FiniteElementSpace nodal_fes(mesh, &fec, mesh->SpaceDimension());
  mesh->SetNodalFESpace(&nodal_fes);

  for (int i = 0; i < refine; ++i) {
    mesh->UniformRefinement();
    snap_nodes(mesh);
  }

  ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);

  cout << "number of nodes: " << pmesh->GetNodes()->Size()/3 << endl;

  delete mesh;

  // Set up ...

  ParFiniteElementSpace fes(pmesh, &fec);

  ParBilinearForm *a = new ParBilinearForm(&fes);
  a->AddDomainIntegrator(new DiffusionIntegrator(one));
  a->AddDomainIntegrator(new MassIntegrator(mu));
  a->Assemble();
  a->Finalize();

  ParBilinearForm *m = new ParBilinearForm(&fes);
  m->AddDomainIntegrator(new MassIntegrator(one));
  m->Assemble();
  m->Finalize();

  HypreParMatrix *A = a->ParallelAssemble();
  HypreParMatrix *M = m->ParallelAssemble();

  HypreBoomerAMG *amg = new HypreBoomerAMG(*A);
  amg->SetPrintLevel(0);

  HypreLOBPCG *lobpcg = new HypreLOBPCG(MPI_COMM_WORLD);
  lobpcg->SetNumModes(nev);
  lobpcg->SetRandomSeed(seed);
  lobpcg->SetPreconditioner(*amg);
  lobpcg->SetMaxIter(200);
  lobpcg->SetTol(1e-8);
  lobpcg->SetPrecondUsageMode(1);
  lobpcg->SetPrintLevel(1);
  lobpcg->SetMassMatrix(*M);
  lobpcg->SetOperator(*A);

  lobpcg->Solve();

  ParGridFunction x(&fes);
  x = lobpcg->GetEigenvector(nev - 1);

  x.Save("x");

  pmesh->GetNodes()->Save("nodes");
}

void snap_nodes(Mesh *mesh) {
  GridFunction &nodes = *mesh->GetNodes();

  Vector node(mesh->SpaceDimension());

  for (int i = 0; i < nodes.FESpace()->GetNDofs(); ++i) {
    for (int d = 0; d < mesh->SpaceDimension(); ++d)
      node(d) = nodes(nodes.FESpace()->DofToVDof(i, d));

    node /= node.Norml2();

    for (int d = 0; d < mesh->SpaceDimension(); ++d)
      nodes(nodes.FESpace()->DofToVDof(i, d)) = node(d);
  }

  if (mesh->Nonconforming()) {
    Vector tnodes;
    nodes.GetTrueDofs(tnodes);
    nodes.SetFromTrueDofs(tnodes);
  }
}
