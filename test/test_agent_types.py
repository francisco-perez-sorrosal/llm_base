from pathlib import Path
import tempfile
from llm_foundation import logger
from llm_foundation.agent_types import Persona, Role

def test_persona_to_yaml():
    role1 = Role(name="role1", description="Role 1 Description", agent_system_message="Agent System Message")
    role2 = Role(name="role2", description="Role 2 Description", agent_system_message="Agent System Message again")
    roles_dict= {"role1": role1, "role2": role2}
    persona = Persona(name="Test Persona", roles=roles_dict)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        
        filename="test_persona.yaml"

        persona.to_yaml_file(Path(temp_dir), filename)
        
        same_persona = Persona.from_yaml_file(str(Path(temp_dir, filename)))

        # Assertions
        assert persona.name == same_persona.name, "Not the same name"
        assert len(persona.roles) == len(same_persona.roles), "Not the same number of roles"
        
        for role_name, role in persona.roles.items():
            same_role = same_persona.roles.get(role_name)
            assert same_role is not None, f"Role {role_name} not found in deserialized persona"
            assert role.name == same_role.name, f"Role name mismatch for {role_name}"
            assert role.description == same_role.description, f"Role description mismatch for {role_name}"
            assert role.agent_system_message == same_role.agent_system_message, f"Agent system message mismatch for {role_name}"