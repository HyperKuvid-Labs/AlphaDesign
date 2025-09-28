import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Calendar, Users, Target, Trophy, Wrench, AlertCircle, CheckCircle2 } from 'lucide-react';
import projectDetails from '@/data/projectDetails.json';

const MilestoneDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();

  // Get milestone data from JSON
  const milestone = projectDetails[id as keyof typeof projectDetails];

  if (!milestone) {
    return (
      <div className="min-h-screen bg-background py-20">
        <div className="container mx-auto px-4 text-center">
          <h1 className="text-4xl font-rajdhani font-bold text-foreground mb-4">
            Milestone Not Found
          </h1>
          <p className="text-muted-foreground mb-8">
            The requested milestone could not be found.
          </p>
          <button 
            onClick={() => navigate('/progress')}
            className="btn-racing"
          >
            Back to Progress
          </button>
        </div>
      </div>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-400 border-green-400 bg-green-400/10';
      case 'in-progress': return 'text-secondary border-secondary bg-secondary/10';
      case 'upcoming': return 'text-muted-foreground border-muted bg-muted/10';
      default: return 'text-muted-foreground border-muted bg-muted/10';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle2 className="w-5 h-5" />;
      case 'in-progress': return <Wrench className="w-5 h-5" />;
      case 'upcoming': return <AlertCircle className="w-5 h-5" />;
      default: return <AlertCircle className="w-5 h-5" />;
    }
  };

  return (
    <div className="min-h-screen bg-background py-20">
      <div className="container mx-auto px-4">
        {/* Back Button */}
        <button 
          onClick={() => navigate('/progress')}
          className="flex items-center space-x-2 text-muted-foreground hover:text-foreground transition-racing mb-8"
        >
          <ArrowLeft className="w-5 h-5" />
          <span className="font-rajdhani font-medium">Back to Progress</span>
        </button>

        {/* Header */}
        <div className="mb-12">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
            <div className="mb-4 md:mb-0">
              <h1 className="text-4xl md:text-6xl font-rajdhani font-bold text-foreground mb-2">
                {milestone.title}
              </h1>
              <div className="flex items-center space-x-4 text-muted-foreground">
                {/* <div className="flex items-center space-x-2">
                  <Calendar className="w-4 h-4" />
                  <span>{milestone.date}</span>
                </div> */}
                {/* <div className="flex items-center space-x-2">
                  <span>Duration: {milestone.duration}</span>
                </div> */}
              </div>
            </div>
            
            {/* Status Badge */}
            {/* <div className={`inline-flex items-center space-x-2 px-4 py-2 rounded-full text-sm font-medium border ${getStatusColor(milestone.status)}`}>
              {getStatusIcon(milestone.status)}
              <span className="capitalize font-rajdhani">{milestone.status.replace('-', ' ')}</span>
            </div> */}
          </div>

          <p className="text-xl font-inter text-muted-foreground leading-relaxed max-w-4xl">
            {milestone.detailedDescription}
          </p>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 mb-12">
          {/* Images Section */}
          <div className="space-y-6">
            <h2 className="text-2xl font-rajdhani font-semibold text-foreground mb-4">
              Project Image
            </h2>
            <div className="grid grid-cols-1 gap-4">
              {milestone.images.map((image, index) => (
                <div key={index} className="relative overflow-hidden rounded-lg">
                  <img 
                    src={image} 
                    alt={`${milestone.title} - Image ${index + 1}`}
                    className="w-full h-auto object-contain hover:scale-105 transition-racing"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent" />
                </div>
              ))}
            </div>
          </div>

          {/* Details Section */}
          <div className="space-y-6">
            <h2 className="text-2xl font-rajdhani font-semibold text-foreground mb-4">
              Project Details
            </h2>
            {/* Objectives */}
            <div className="racing-card p-6">
              <div className="flex items-center space-x-2 mb-4">
                <Target className="w-5 h-5 text-primary" />
                <h3 className="text-xl font-rajdhani font-semibold text-foreground">
                  Objectives
                </h3>
              </div>
              <ul className="space-y-2">
                {milestone.objectives.map((objective, index) => (
                  <li key={index} className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-primary mt-2 flex-shrink-0"></div>
                    <span className="text-muted-foreground font-inter">{objective}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Achievements */}
            <div className="racing-card p-6">
              <div className="flex items-center space-x-2 mb-4">
                <Trophy className="w-5 h-5 text-green-400" />
                <h3 className="text-xl font-rajdhani font-semibold text-foreground">
                  Key Achievements
                </h3>
              </div>
              <ul className="space-y-2">
                {milestone.achievements.map((achievement, index) => (
                  <li key={index} className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-green-400 mt-2 flex-shrink-0"></div>
                    <span className="text-muted-foreground font-inter">{achievement}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Team Members
            <div className="racing-card p-6">
              <div className="flex items-center space-x-2 mb-4">
                <Users className="w-5 h-5 text-secondary" />
                <h3 className="text-xl font-rajdhani font-semibold text-foreground">
                  Team Members
                </h3>
              </div>
              <div className="grid grid-cols-1 gap-2">
                {milestone.teamMembers.map((member, index) => (
                  <div key={index} className="text-muted-foreground font-inter">
                    {member}
                  </div>
                ))}
              </div>
            </div> */}
          </div>
        </div>

        {/* Technologies & Tools
        <div className="mb-12">
          <h2 className="text-2xl font-rajdhani font-semibold text-foreground mb-6">
            Technologies & Tools Used
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {milestone.technologies.map((tech, index) => (
              <div key={index} className="racing-card p-4 text-center">
                <span className="text-muted-foreground font-inter">{tech}</span>
              </div>
            ))}
          </div>
        </div> */}

        {/* Challenges & Outcomes */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Challenges */}
          <div className="racing-card p-6">
            <h3 className="text-xl font-rajdhani font-semibold text-foreground mb-4">
              Challenges Faced
            </h3>
            <ul className="space-y-3">
              {milestone.challenges.map((challenge, index) => (
                <li key={index} className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-red-400 mt-2 flex-shrink-0"></div>
                  <span className="text-muted-foreground font-inter">{challenge}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Outcomes */}
          <div className="racing-card p-6">
            <h3 className="text-xl font-rajdhani font-semibold text-foreground mb-4">
              Key Outcomes
            </h3>
            <ul className="space-y-3">
              {milestone.outcomes.map((outcome, index) => (
                <li key={index} className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-blue-400 mt-2 flex-shrink-0"></div>
                  <span className="text-muted-foreground font-inter">{outcome}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MilestoneDetail;